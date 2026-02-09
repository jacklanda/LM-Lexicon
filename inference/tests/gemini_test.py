import google.generativeai as genai


genai.configure(api_key="AIzaSyDjbyS0TRx-Yr1YsmWY80nj2KUAM0lZSfk")

# list available models
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

model = genai.GenerativeModel("gemini-pro")

prompt = "Write a story about a magic backpack."

response = model.generate_content(prompt)

print(response.text)
