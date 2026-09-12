#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    role: Role,
    content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[cfg(test)]
mod tests {
    use super::{Message, Role};

    #[test]
    fn role_string_matches_chat_role_names() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    #[test]
    fn creates_user_message() {
        let message = Message::user("Hello");

        assert_eq!(message.role(), Role::User);
        assert_eq!(message.content(), "Hello");
    }

    #[test]
    fn creates_system_message() {
        let message = Message::system("You are a helpful assistant.");

        assert_eq!(message.role(), Role::System);
        assert_eq!(message.content(), "You are a helpful assistant.");
    }

    #[test]
    fn creates_assistant_message() {
        let message = Message::assistant("Hello!");

        assert_eq!(message.role(), Role::Assistant);
        assert_eq!(message.content(), "Hello!");
    }
}
