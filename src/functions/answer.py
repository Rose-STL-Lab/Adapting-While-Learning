function_answer = [
    {
        "type": "function",
        "function": {
            "name": "answer_question",
            "description": "Answer the Question.",
            "parameters": {
                "type": "object",
                "properties": {
                    # f(., ., .)
                    "thought": {
                        "type": "string",
                        "description": "Internal reasoning and thoughts of why you call this function.",
                    },
                    "sloving_process": {
                        "type": "string",
                        "description": "Detailed list how do you solve this question, step by step. If you wrote code and got result from it, you should write how the problem was solved based on the output of the code, but don't mention your coding here.",
                    },
                    "answer": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "Your answer to this question. If you have multiple answers, you can write them all. If none of the answers are correct, you can give your answer as well.",
                    },
                },
                "required": ["thought", "sloving_process", "answer"],
            },
        },
    }
]

function_answer_open = [
    {
        "type": "function",
        "function": {
            "name": "answer_question",
            "description": "Answer the Question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Internal reasoning and thoughts of why you call this function.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "Your answer to this question. It should be a list of locations. If you are going to propose a maritime route, it starts from the given start point and ends at the given ending point. If you are going to propose a transfer station, if is a 3 element list, the first element is the start point, the second element is the transfer station, and the third element is the ending point.",
                    },
                    "sloving_process": {
                        "type": "string",
                        "description": "Detailed list how do you solve this question, step by step. If you use tools and got result from it, you should write how the problem was solved based on the output of the code, but don't mention your tool using here.",
                    },
                },
                "required": ["thought", "answer", "sloving_process"],
            },
        },
    }
]
