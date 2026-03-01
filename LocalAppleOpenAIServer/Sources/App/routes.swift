import Vapor

func routes(_ app: Application) throws {
    app.get { req async in
        "Local Apple OpenAI Server is running!"
    }

    if #available(macOS 26.0, *) {
        let chatController = ChatController()
        
        // OpenAI-compatible endpoints
        app.post("v1", "chat", "completions", use: chatController.createChatCompletion)

        // Internal summarization endpoint for context condensation
        app.post("v1", "summarize", use: chatController.summarize)
        } else {
        app.logger.critical("FoundationModels is not available on this macOS version. Upgrade to 26.0+.")
    }
}
