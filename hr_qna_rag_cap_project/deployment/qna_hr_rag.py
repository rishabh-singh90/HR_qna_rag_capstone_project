
class RAGResponseGenerator:

    qna_system_message = """
        You are an AI Human Resource(HR) specialist of a company. As an HR you will be asked queries related to company's HR and employee policies.
        User input will have the context required by you to answer user questions.

        The context will begin with the token: ###Context.
        The context contains references to specific portions of a document relevant to the user query. Your task is to review this context and provide the appropriate answers from the context.

        User questions will begin with the token: ###Question.
        Provide and elaborate answer by using ONLY the context provided in the input. Do not mention anything about the context in your final answer.

        Answer your question in bullet points and provide elborate answer.
        Do not give extra details which are not related to queries.
    """

    qna_user_message_template = """
    ###Context
    Here are some context that are relevant to the question mentioned below.
    {context}

    ###Question
    {question}
    """

    # Constructor method to initialize object attributes
    def __init__(self, retriever, lcpp_llm):
        self.retriever = retriever  # Instance attribute retriever
        self.lcpp_llm = lcpp_llm  # Instance attribute lcpp_llm

    def generate_rag_response(self,user_input,k=2,max_tokens=512,temperature=0,top_p=0.95,top_k=2):
        
        # Retrieve relevant document chunks
        relevant_document_chunks = self.retriever.get_relevant_documents(query=user_input,k=k)
        context_list = [d.page_content for d in relevant_document_chunks]

        # Combine document chunks into a single context for query
        context_for_query = ". ".join(context_list)

        user_message = RAGResponseGenerator.qna_user_message_template.replace('{context}', context_for_query)
        user_message = user_message.replace('{question}', user_input)

        # Combine user_prompt and system_message to create the prompt
        prompt = f"""[INST]{RAGResponseGenerator.qna_system_message}\n
                    {'user'}: {user_message} \n
                    [/INST]"""

        # Generate the response
        try:
            response = self.lcpp_llm(
                      prompt=prompt,
                      max_tokens=max_tokens,
                      temperature=temperature,
                      top_p=top_p,
                      top_k=top_k,
                      stop=['INST'],
                      repeat_penalty=1.2
                      )

            # Extract and print the model's response
            response = response['choices'][0]['text'].strip()
        except Exception as e:
            response = f'Sorry, I encountered the following error: \n {e}'

        return response
