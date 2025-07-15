from jload import jload, jsave
from vlllm import generate
import random

instruction = """
You are a teacher specialized in climate science and data analysis. Your task is to provide a detailed solution process for a given questions about climate issues. The question was answered with relevant tools and logical reasoning, but you shouldn't mention these tools using procedures in your answer. The solution should end with 'the answer is A/B/C/D'.

You should provide a solution process that includes:
- Relevant data from the climate simulator (but you shouldn't mention the simulations and simulators, you should directly list relevant data)
- Detailed analysis of this data
- Logical reasoning connecting the data to the question

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why.

Example 1:
Question: How much will the temperature of Chitré in 2057 under ssp585 change if the emission of CH4 is increased by 65%?
Options:
A. 0.150196
B. 0.2713890075683594
C. 0.203833
D. 0.018829
Calling query_lat_and_lon with arguments: {'city_name': 'Chitré', 'thought': 'I need to find the latitude and longitude of Chitré to retrieve the temperature information.'}
Feedback from query_lat_and_lon: latitude: 7.9667, longitude: -80.4333.
Calling future_temperature with arguments: {'longitude': -80.4333, 'latitude': 7.9667, 'year': 2057, 'setting': 'ssp585', 'thought': 'I want to check the predicted temperature in 2057 under ssp585 without any CH4 emission increase.'}
Feedback from future_temperature: The temperature is 29.478918075561523.
Calling diy_greenhouse with arguments: {'longitude': -80.4333, 'latitude': 7.9667, 'setting': 'ssp585', 'year': 2057, 'delta_CO2': 0, 'delta_CH4': 0.65, 'thought': 'I will retrieve the temperature with a 65% increase in CH4 emissions.'}
Feedback from diy_greenhouse: The temperature is 29.750307083129883.
Calling answer_question with arguments: {'thought': 'To find the change in temperature due to a 65% increase in CH4 emissions, I subtracted the original predicted temperature from the new temperature with increased CH4. The change is 29.750307083129883 - 29.478918075561523 = 0.2713890075683594.', 'answer': 'B', 'sloving_process': '1. Retrieved the latitude and longitude for Chitré.\n2. Fetched the predicted temperature for Chitré in 2057 under the ssp585 scenario without CH4 emission changes.\n3. Used the diy_greenhouse function to predict the temperature with a 65% increase in CH4 emissions.\n4. Calculated the change in temperature by subtracting the original temperature from the new temperature with increased emissions.\n5. Compared the calculated change to the given options and selected the correct answer.'}

Answer:
Chitré is located at latitude 7.9667 and longitude -80.4333. Based on climate projections for this location, the average temperature in 2057 under the ssp585 scenario is expected to be 29.478918°C if there is no change in methane (CH₄) emissions. If CH₄ emissions are increased by 65%, the predicted temperature for the same year and scenario rises to 29.750307°C. This means that a 65% increase in methane emissions leads to an additional warming of 0.271389°C for Chitré in 2057. The reasoning behind this result is that methane is a potent greenhouse gas, and such a significant increase in its atmospheric concentration intensifies the greenhouse effect, trapping more heat and thus raising the average temperature. After comparing the difference to the provided options, it is clear that the temperature change is 0.2713890075683594. Therefore, the answer is B.

Example 2:

Question: For Ebreichsdorf, Gleisdorf, Perchtoldsdorf, Voitsberg, which city has the lowest temperature in 2058 under ssp245?
Options:
A. Perchtoldsdorf
B. Ebreichsdorf
C. Gleisdorf
D. Voitsberg
Calling query_lat_and_lon with arguments: {'city_name': 'Ebreichsdorf', 'thought': 'I need the latitude and longitude to retrieve future temperature data for Ebreichsdorf.'}
Feedback from query_lat_and_lon: latitude: 47.9667, longitude: 16.4.
Calling query_lat_and_lon with arguments: {'city_name': 'Gleisdorf', 'thought': 'I need the latitude and longitude to retrieve future temperature data for Gleisdorf.'}
Feedback from query_lat_and_lon: latitude: 47.1039, longitude: 15.7083.
Calling query_lat_and_lon with arguments: {'city_name': 'Perchtoldsdorf', 'thought': 'I need the latitude and longitude to retrieve future temperature data for Perchtoldsdorf.'}
Feedback from query_lat_and_lon: latitude: 48.1167, longitude: 16.2667.
Calling query_lat_and_lon with arguments: {'city_name': 'Voitsberg', 'thought': 'I need the latitude and longitude to retrieve future temperature data for Voitsberg.'}
Feedback from query_lat_and_lon: latitude: 47.0333, longitude: 15.15.
Calling future_temperature with arguments: {'longitude': 16.4, 'latitude': 47.9667, 'year': 2058, 'setting': 'ssp245', 'thought': 'Retrieve future temperature for Ebreichsdorf.'}
Feedback from future_temperature: The temperature is 9.03313159942627.
Calling future_temperature with arguments: {'longitude': 15.7083, 'latitude': 47.1039, 'year': 2058, 'setting': 'ssp245', 'thought': 'Retrieve future temperature for Gleisdorf.'}
Feedback from future_temperature: The temperature is 9.273866653442383.
Calling future_temperature with arguments: {'longitude': 16.2667, 'latitude': 48.1167, 'year': 2058, 'setting': 'ssp245', 'thought': 'Retrieve future temperature for Perchtoldsdorf.'}
Feedback from future_temperature: The temperature is 8.793405532836914.
Calling future_temperature with arguments: {'longitude': 15.15, 'latitude': 47.0333, 'year': 2058, 'setting': 'ssp245', 'thought': 'Retrieve future temperature for Voitsberg.'}
Feedback from future_temperature: The temperature is 9.41732406616211.
Calling answer_question with arguments: {'thought': 'I have retrieved the future temperatures for all the cities in 2058 under the ssp245 scenario. By comparing the temperatures, I can determine the city with the lowest temperature.', 'answer': 'A', 'sloving_process': '1. Retrieved the latitude and longitude for each city: Ebreichsdorf, Gleisdorf, Perchtoldsdorf, and Voitsberg.\n2. Retrieved the future temperature for each city in 2058 under the ssp245 scenario:\n   - Ebreichsdorf: 9.03°C\n   - Gleisdorf: 9.27°C\n   - Perchtoldsdorf: 8.79°C\n   - Voitsberg: 9.42°C\n3. Compared the temperatures and found that Perchtoldsdorf has the lowest temperature at 8.79°C.\n4. Selected Perchtoldsdorf as the answer.'}

Answer:

Ebreichsdorf, located at latitude 47.9667 and longitude 16.4, is projected to have an average temperature of 9.03°C in 2058 under the ssp245 scenario. Gleisdorf sits at latitude 47.1039, longitude 15.7083, and its projected temperature for the same year and scenario is 9.27°C. Perchtoldsdorf, at latitude 48.1167 and longitude 16.2667, is expected to have a lower temperature of 8.79°C, while Voitsberg, at latitude 47.0333 and longitude 15.15, is projected to reach 9.42°C. 

Comparing these values, Perchtoldsdorf stands out as having the lowest projected average temperature in 2058 among the four cities. This result is based on the direct comparison of the temperature data for each city's coordinates under the same climate scenario and year. Therefore, the answer is A.

Now Begin!
"""

def process_entry(item):
    """
    Process a single entry to extract the prompt and question.
    """
    if "/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct" not in item or not item["/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"]:
        return None
    messages = item["/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"]
    prompt = "Following is the interaction for answering the question.\n\n"
    for message in messages:
        if message["role"] == "user":
            prompt += message["content"]
        elif message["role"] == "assistant":
            prompt += f"\nCalling function: `{message['content']}."
        elif message["role"] == "function":
            prompt += f"\nFeedback from `{message['name']}`: {message['content']}."

    prompt += "Following is the question that you should write solution for.\n\n"

    problem_text = f"Question: {item['Question']}\nOptions:\nA. {item['Options'][0]}\nB. {item['Options'][1]}\nC. {item['Options'][2]}\nD. {item['Options'][3]}"

    return prompt + problem_text + "Now begin to write solution for this question based on the data, question and answer."



if __name__ == "__main__":
    data = jload("climate_train.json")

    random.shuffle(data)

    for d in data:
        d["input"] = process_entry(d)

    processed_data = generate(
        model_id="/home/test/test12/bohan/models/Qwen2.5-14B-Instruct",
        data=[item.copy() for item in data if item["input"]],
        system=instruction,
        message_key="input",
        tp=2,
        pp=1,
        n=1,
        worker_num=4,
        temperature=0.1,
        use_sample=False,
        result_key="solution",
        max_model_len=8000,
        max_output_len=4000,
        chunk_size=1000,
        gpu_assignments=None,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        dtype="auto"
    )

    jsave(processed_data, "climate_train.json")