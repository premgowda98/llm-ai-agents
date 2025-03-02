## Agents

1. [History Retriever]("https://dev.to/guilhermecxe/how-a-history-aware-retriever-works-5e07")
2. [Conversational Retrieval Chain]("https://vijaykumarkartha.medium.com/beginners-guide-to-conversational-retrieval-chain-using-langchain-3ddf1357f371")
3. Agent Executor
    1. Decides which tool has to be used and in which order
    2. Accepts agents and tools
3. LLMSingleActionAgent
    1. Executes a single action 
4. AgentOutPutParser
    1. To Parse output from LLM
5. AgentExecutor is like a orchestrator
    1. It invokes the LLMSingleActionAgent
    2. Get the response and decides which tool to use next

## Memory

1. ConversationalBuffer -> Stores entire conversation
2. ConversationSummary -> Stores summary of prev conversation
3. ConversationalBufferWindow -> sliding window
4. ConversationKnowledgeGraph -> 