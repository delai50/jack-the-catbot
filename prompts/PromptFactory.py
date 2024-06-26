

class PromptFactory:

    question_answer_template = """
                                You are an expert of world knowledge. I am going to ask you a question. 
                                Your response should be comprehensive and not contradicted with the following 
                                context if they are relevant. Otherwise, ignore them if they are not relevant.
                                Do not hallucinate, if the context is not relevant just say that you dont know the answer.
                                Try to generate a numbered list or bullet point list as output if you think is possible.
                                Only response if given context.
                                Ok, now take a deep breath and do the task step by step.

                                # Context: {context}
                                
                                # Original Question: {question}
                                
                                # Answer:
                                """


    system_router_template = """
                            Your role is to accurately classify user queries into the most appropriate category:
                            questionanswer, python, or other, even if they contain typos or slight variations.
            
                            For a given input, you need to output a single token `questionanswer`, `python`, or `other`.
                            Do not respond with more than one word. If you doubt between `questionanswer` or `other`,
                            priorizite `questionanswer`.
            
                            Choose the proper class based on the following definitions:
                            
                            questionanswer: Suitable for queries that can be answered without performing any additional actions.
                            
                            python: Appropiate for consults that require the execution of python code to answer the question.
                            
                            other: Applied to queries that do not fit into the specified categories.
            
                            EXAMPLES:
            
                            Question: 'What is Microsoft total revenue in 2022?'
                            questionanswer
            
                            Question: If I have 3 apples and I eat 2, how many apples are left?
                            python
            
                            Question: Thank you!
                            other
                            """

    neutral_template = """
                       You are an assistant and your task is to help user.
                       """        