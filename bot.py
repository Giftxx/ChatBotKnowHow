# code testu
from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import TextSendMessage, TemplateSendMessage, ButtonsTemplate, MessageAction
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import ollama
from functools import lru_cache

# การตั้งค่า Neo4j
URI = "neo4j://localhost"
AUTH = ("neo4j", "GIft_4438")

# ตรวจสอบการเชื่อมต่อกับ Neo4j
driver = None
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Connected to Neo4j successfully.")
except Exception as e:
    print("Failed to connect to Neo4j:", str(e))

# สร้างโมเดล SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ดึงข้อมูลจาก Neo4j
cypher_query = '''
MATCH (n:Greeting) 
RETURN n.name as name, n.msg_reply as reply
UNION ALL
MATCH (n:Question) 
RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
greeting_replies = []

if driver:
    try:
        with driver.session() as session:
            results = session.run(cypher_query)
            for record in results:
                greeting_corpus.append(record['name'])
                greeting_replies.append(record['reply'])
    except Exception as e:
        print("Failed to run query:", str(e))
   
# ใช้ข้อมูลทดสอบถ้าไม่พบข้อมูลในฐานข้อมูล
if len(greeting_corpus) == 0:
    greeting_corpus = ["สวัสดี", "ดีจ้า", "สวัสดีครับ"]
    greeting_replies = ["สวัสดีครับ", "ดีจ้า", "สวัสดีครับ"]

# สร้าง embedding vectors
if len(greeting_corpus) > 0:
    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
else:
    greeting_vec = None

# ตั้งค่า Ollama Client
ollama_client = ollama.Client()

# ฟังก์ชันที่ใช้แคชสำหรับคำตอบจาก Ollama
@lru_cache(maxsize=100)
def generate_ollama_response(sentence, language):
    if language == 'TH':
        prompt = f"{sentence} ขอคำตอบแบบกระชับไม่เกิน 50 คำ"
    else:
        prompt = f"{sentence} Please provide a concise answer within 50 words."
    
    try:
        response = ollama_client.generate(
            model="llama3.1:latest",
            prompt=prompt
        )
        return response.get('response', "Sorry, I couldn't find an answer.")
    except Exception as ollama_error:
        return f"Error in Ollama: {str(ollama_error)}"

def compute_response(sentence, language):
    if not sentence:
        return "ขอโทษครับ ไม่พบข้อความในคำถามนี้"

    if language == 'TH':
        if greeting_vec is None:
            return "ขอโทษครับ ไม่สามารถให้บริการได้ในขณะนี้"

        ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
        
        try:
            greeting_scores = util.cos_sim(greeting_vec, ask_vec)  
            greeting_np = greeting_scores.cpu().numpy()
            max_index = np.argmax(greeting_np)
            max_score = greeting_np[max_index]

            if max_score > 0.6:
                response_msg = greeting_replies[max_index] + " [ตอบจาก Neo4j]"
            else:
                response_msg = generate_ollama_response(sentence, language) + " [ตอบจาก Ollama]"

        except Exception as e:
            response_msg = f"เกิดข้อผิดพลาดในการคำนวณ: {str(e)}"
        
        return response_msg
    else:
        return generate_ollama_response(sentence, language) + " [ตอบจาก Ollama]"

app = Flask(__name__)

# ฟังก์ชันการเลือกภาษา
language_selection = {}

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)                    
    try:
        json_data = json.loads(body)                        
        access_token = '8PaaGz+J0J1mm4VefTzK7ZaUPX+YR7fll5DqhCQaUtkwBNjOMjFNey+Ol+lHNvR4T3A9abZLWJub1407ZWqzP0/CGzVPEGVaKQQA7YuH3Rqq+uIcqlNX3VlioTnWXh1PTHN8VtCjQhjSq7o6pi+EIgdB04t89/1O/w1cDnyilFU='
        secret = '88fdadd23cbb53a21e351a930a30af5f'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']

        handler.handle(body, signature)

        if json_data.get('events'):
            event = json_data['events'][0]
            user_id = event['source']['userId']
            msg = event['message']['text']      
            tk = event['replyToken']            

            # หากเป็นครั้งแรกที่ผู้ใช้เข้ามาให้แสดงตัวเลือกภาษา
            if user_id not in language_selection:
                # แสดงการเลือกภาษาทันที
                buttons_template = ButtonsTemplate(
                    title='🌐 Select Language 🌐',
                    text='Please select a language 🗣️',
                    actions=[
                        MessageAction(label="🇹🇭 ภาษาไทย", text="TH"),
                        MessageAction(label="🇬🇧 English", text="ENG")
                    ]
                )
                template_message = TemplateSendMessage(
                    alt_text='Language Selection',
                    template=buttons_template
                )
                line_bot_api.reply_message(tk, template_message)
                language_selection[user_id] = None

            elif msg.lower() == "เลือกภาษา":
                # ให้ผู้ใช้เลือกภาษาใหม่
                buttons_template = ButtonsTemplate(
                    title='🌐 Select Language 🌐',
                    text='Please select a language 🗣️',
                    actions=[
                        MessageAction(label="🇹🇭 ภาษาไทย", text="TH"),
                        MessageAction(label="🇬🇧 English", text="ENG")
                    ]
                )
                template_message = TemplateSendMessage(
                    alt_text='Language Selection',
                    template=buttons_template
                )
                language_selection[user_id] = None
                line_bot_api.reply_message(tk, template_message)

            elif user_id in language_selection and language_selection[user_id] is None:
                # บันทึกภาษาที่ผู้ใช้เลือก
                if msg.lower() == "th":
                    language_selection[user_id] = 'TH'
                    response_msg = "คุณเลือกภาษาไทยแล้ว"
                elif msg.lower() == "eng":
                    language_selection[user_id] = 'ENG'
                    response_msg = "You have selected English."
                else:
                    response_msg = "กรุณาเลือกภาษา: TH หรือ ENG"

                text_message = TextSendMessage(text=response_msg)
                line_bot_api.reply_message(tk, text_message)

            else:
                # ตอบกลับข้อความตามภาษาที่ผู้ใช้เลือก
                response_msg = compute_response(msg, language_selection[user_id])
                text_message = TextSendMessage(text=response_msg)
                line_bot_api.reply_message(tk, text_message)
                print(msg, tk)                                      

        else:
            return 'No events to process', 400

    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        return 'Invalid signature', 400
    except Exception as e:
        print("An error occurred:", str(e))
        print("Request Body:", body)
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)
