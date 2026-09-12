use super::Message;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Conversation {
    messages: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system(content: impl Into<String>) -> Self {
        let mut conversation = Self::new();
        conversation.push_system(content);
        conversation
    }

    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn push_system(&mut self, content: impl Into<String>) {
        self.push(Message::system(content));
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.push(Message::user(content));
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.push(Message::assistant(content));
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn last(&self) -> Option<&Message> {
        self.messages.last()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::Conversation;
    use crate::chat::{Message, Role};

    #[test]
    fn new_conversation_is_empty() {
        let conversation = Conversation::new();

        assert!(conversation.is_empty());
        assert_eq!(conversation.len(), 0);
        assert!(conversation.last().is_none());
    }

    #[test]
    fn creates_conversation_with_system_message() {
        let conversation = Conversation::with_system("You are a helpful assistant.");

        assert_eq!(conversation.len(), 1);

        let message = conversation.last().unwrap();

        assert_eq!(message.role(), Role::System);
        assert_eq!(message.content(), "You are a helpful assistant.");
    }

    #[test]
    fn appends_messages_in_order() {
        let mut conversation = Conversation::new();

        conversation.push_system("You are helpful.");
        conversation.push_user("Hello");
        conversation.push_assistant("Hi!");
        conversation.push_user("What did I say?");

        assert_eq!(conversation.len(), 4);

        assert_eq!(conversation.messages()[0].role(), Role::System);

        assert_eq!(conversation.messages()[1].role(), Role::User);

        assert_eq!(conversation.messages()[1].content(), "Hello");

        assert_eq!(conversation.messages()[2].role(), Role::Assistant);

        assert_eq!(conversation.messages()[3].role(), Role::User);
    }

    #[test]
    fn accepts_existing_message() {
        let mut conversation = Conversation::new();

        conversation.push(Message::user("Hello"));

        assert_eq!(conversation.len(), 1);
        assert_eq!(conversation.messages()[0], Message::user("Hello"));
    }

    #[test]
    fn creates_from_existing_messages() {
        let conversation = Conversation::from_messages(vec![
            Message::system("Be concise."),
            Message::user("Hello"),
        ]);

        assert_eq!(conversation.len(), 2);
        assert_eq!(conversation.messages()[0].role(), Role::System);
        assert_eq!(conversation.messages()[1].role(), Role::User);
    }

    #[test]
    fn clears_messages() {
        let mut conversation = Conversation::with_system("System");

        conversation.push_user("Hello");

        assert_eq!(conversation.len(), 2);

        conversation.clear();

        assert!(conversation.is_empty());
    }
}
