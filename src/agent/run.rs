use crate::{
    agent::{
        AgentConfig, AgentDecision, AgentStep, AgentTrace, build_agent_decision_prompt,
        build_agent_final_prompt, parse_agent_decision,
    },
    chat::{ChatTemplate, Conversation, generate_chat_stream},
    error::{Result, TinyError},
    generation::{GenerationConfig, GenerationOutput},
    metal::MetalContext,
    model::Transformer,
    structured::StructuredRetryConfig,
    tokenizer::Tokenizer,
    tool_calling::generate_tool_arguments_for_tool,
    tools::{ToolCall, ToolRegistry},
};

#[derive(Debug)]
pub struct AgentOutput {
    answer: String,
    trace: AgentTrace,
    generation: GenerationOutput,
}

impl AgentOutput {
    pub fn answer(&self) -> &str {
        &self.answer
    }

    pub fn trace(&self) -> &AgentTrace {
        &self.trace
    }

    pub fn generation(&self) -> &GenerationOutput {
        &self.generation
    }

    pub fn into_parts(self) -> (String, AgentTrace, GenerationOutput) {
        (self.answer, self.trace, self.generation)
    }
}

pub fn run_agent(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    registry: &ToolRegistry,
    user_request: &str,
    agent_config: &AgentConfig,
    generation_config: &GenerationConfig,
    retry_config: &StructuredRetryConfig,
) -> Result<AgentOutput> {
    if registry.is_empty() {
        return Err(TinyError::Tool(
            "agent requires at least one registered tool".to_string(),
        ));
    }

    let user_request = user_request.trim();

    if user_request.is_empty() {
        return Err(TinyError::InvalidArgument(
            "agent request cannot be empty".to_string(),
        ));
    }

    let mut trace = AgentTrace::new();

    for step_index in 0..agent_config.max_steps() {
        /*
         * ------------------------------------
         * 1. Decide next action
         * ------------------------------------
         */
        let decision = generate_agent_decision(
            context,
            model,
            tokenizer,
            template,
            conversation,
            registry,
            user_request,
            &trace,
            generation_config,
        )?;

        match decision {
            AgentDecision::Final => {
                return generate_final_answer(
                    context,
                    model,
                    tokenizer,
                    template,
                    conversation,
                    user_request,
                    trace,
                    generation_config,
                );
            }

            AgentDecision::Tool(tool_name) => {
                /*
                 * ------------------------------------
                 * 2. Resolve selected Tool
                 * ------------------------------------
                 */
                let tool = registry.get(&tool_name).ok_or_else(|| {
                    TinyError::Tool(format!("agent selected unknown tool `{tool_name}`"))
                })?;

                /*
                 * ------------------------------------
                 * 3. Generate arguments ONLY
                 * for the already-selected Tool.
                 * ------------------------------------
                 */
                let tool_request = if trace.is_empty() {
                    user_request.to_string()
                } else {
                    let mut req = user_request.to_string();
                    req.push_str("\n\nPrevious tool observations:\n");
                    for step in trace.steps() {
                        req.push_str(&format!("{}: {}\n", step.tool_name(), step.result()));
                    }
                    req
                };

                let generated_arguments = generate_tool_arguments_for_tool(
                    context,
                    model,
                    tokenizer,
                    template,
                    conversation,
                    tool,
                    &tool_request,
                    generation_config,
                    retry_config,
                )?;

                let arguments = generated_arguments.arguments().clone();

                let call = ToolCall::new(tool_name.clone(), arguments.clone());

                /*
                 * ------------------------------------
                 * 4. Execute
                 * ------------------------------------
                 */
                let result = registry.execute(&call)?;

                /*
                 * ------------------------------------
                 * 5. Observation -> Trace
                 * ------------------------------------
                 */
                trace.push(AgentStep::new(
                    step_index,
                    tool_name,
                    arguments,
                    result.output().clone(),
                ));
            }
        }
    }

    /*
     * Model이 max_steps 내에 final을 선택하지 않아도
     * 무한 루프는 허용하지 않는다.
     *
     * 지금까지 모은 observation으로 강제 finalization.
     */
    generate_final_answer(
        context,
        model,
        tokenizer,
        template,
        conversation,
        user_request,
        trace,
        generation_config,
    )
}

fn generate_agent_decision(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    registry: &ToolRegistry,
    user_request: &str,
    trace: &AgentTrace,
    generation_config: &GenerationConfig,
) -> Result<AgentDecision> {
    let prompt = build_agent_decision_prompt(user_request, trace, registry)?;

    let mut turn = conversation.clone();

    turn.push_user(prompt);

    /*
     * Decision은 tool 이름 하나만 필요하다.
     */
    let decision_config = GenerationConfig {
        max_new_tokens: 16,
        temperature: 0.0,
        top_k: None,
        top_p: 1.0,
        seed: generation_config.seed,
    };

    let mut tokens = Vec::new();

    generate_chat_stream(
        context,
        model,
        tokenizer,
        template,
        &turn,
        &decision_config,
        |token_id| {
            tokens.push(token_id);

            Ok(())
        },
    )?;

    let raw = tokenizer.decode(&tokens, true)?.trim().to_string();

    parse_agent_decision(&raw, registry)
}

fn generate_final_answer(
    context: &MetalContext,
    model: &Transformer,
    tokenizer: &dyn Tokenizer,
    template: &dyn ChatTemplate,
    conversation: &Conversation,
    user_request: &str,
    trace: AgentTrace,
    generation_config: &GenerationConfig,
) -> Result<AgentOutput> {
    let prompt = build_agent_final_prompt(user_request, &trace)?;

    let mut turn = conversation.clone();

    turn.push_user(prompt);

    let final_config = GenerationConfig {
        max_new_tokens: generation_config.max_new_tokens.min(64),

        temperature: generation_config.temperature,

        top_k: generation_config.top_k,

        top_p: generation_config.top_p,

        seed: generation_config.seed,
    };

    let mut tokens = Vec::new();

    let generation = generate_chat_stream(
        context,
        model,
        tokenizer,
        template,
        &turn,
        &final_config,
        |token_id| {
            tokens.push(token_id);

            Ok(())
        },
    )?;

    let answer = tokenizer.decode(&tokens, true)?.trim().to_string();

    if answer.is_empty() {
        return Err(TinyError::Tool(
            "agent produced an empty final answer".to_string(),
        ));
    }

    Ok(AgentOutput {
        answer,
        trace,
        generation,
    })
}
