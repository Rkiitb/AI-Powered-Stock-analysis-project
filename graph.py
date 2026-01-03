import streamlit as st
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt,ChatPromptTemplate
import yfinance as yf
from yahooquery import search
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from typing import TypedDict,Dict,Annotated,Literal
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,SystemMessage
from pygooglenews import GoogleNews
from langgraph.checkpoint.memory import MemorySaver  # Or your checkpointer
import asyncio


# env var
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']='Stock analysis project'
load_dotenv()



llm = ChatGroq(model="llama-3.1-8b-instant")  #https://console.groq.com/dashboard/metrics


# state
class StockState(TypedDict):
    user_query:str
    intent:str
    stock_name:str
    ticker:str
    fundamental_data:Dict
    latest_stock_news:str  
    final_result:str


# Intent classifier 
class Intent_classifier(BaseModel):
    intent:Literal['general talk','business news','stock news','buy sell']=Field(...,description='user input classification')

Intent_classifier_llm=llm.with_structured_output(Intent_classifier)


def Intent_finder(state:StockState):
    messages=[
        SystemMessage(content="You are an intent classification assistant. "
                "Your task is to classify the user's query into exactly ONE category."),
        HumanMessage(content=f"""
Classify the user query into one of the following categories:

1. general talk  
   - Casual conversation  
   - Greetings, opinions, explanations, learning questions  
   - No news or trading intent  

2. business news  
   - Daily business updates  
   - Market-wide news (economy, indices, sectors, global markets)  
   - Not about a specific stock  

3. stock news  
   - News related to a specific company or stock  
   - Earnings, results, mergers, announcements, price movement  
   - Example: "What happened to TCS today?"

4. buy sell  
   - User is asking whether they should buy, sell, or hold a stock  
   - Investment or trading advice  
   - Example: "Should I buy Infosys now?"

User query:
{state['user_query']}

Respond ONLY with the category name.
""")
    ]
    response=Intent_classifier_llm.invoke(messages)
    return {'intent':response.intent}
    

# Genral query handler
def General_query_handler(state:StockState):
    result=llm.invoke(state['user_query'])
    return {'final_result':result.content}


# Stock name finder from user query
class stock_name_pydantic(BaseModel):
    stock_name:str


stock_name_llm=llm.with_structured_output(stock_name_pydantic)

def Stock_name_finder(state:StockState):
    message=[
        SystemMessage(content="""You are an helpful assistant. Your job is to find out stock name from give user query. If you dont know the answer then retun empty string"""),
        HumanMessage(content=f""" Find our stock name from given user query. User query : {state['user_query']}""")
    ]

    response=stock_name_llm.invoke(message)
    return {'stock_name':response.stock_name}



# Ticker finder for any stock
class ticker_pydantic(BaseModel):
    ticker:str



def Ticker_finder(state:StockState):
    ticker_name_llm=llm.with_structured_output(ticker_pydantic)
    try:
        results = search(state['stock_name'])            
        matching_symbols={}
        for item in results['quotes']:
            longname = item.get("longname") or item.get("shortname") or "N/A"
            matching_symbols.update({longname:item.get('symbol')})
    except Exception as e:
        return {'ticker':''}

    message=[
        SystemMessage(content="""You are an helpful ai assistant. Your job is to find out most suitable ticker from a provided dictionary.
                     ticker is a code name for any stock. The dictonary contains company name as keys and ticker as values. You have to give me one most suitable ticker for the given stock """),
                     HumanMessage(content=f"""Here is user provided dictionary :{matching_symbols}. You have to find out which stock name is most close to {state['stock_name']} and give me corresponding ticker.""")
    ]

    response=ticker_name_llm.invoke(message)
    return {'ticker':response.ticker}


# Fundamental metrics finder
def Fundamental_finder(state:StockState):
    ticker = yf.Ticker(state['ticker'])

    if not ticker:
        return {'fundamental_data':'No Data available'}



    try:
        # Basic Info
        info = ticker.info or {}
        current_price = ticker.fast_info["last_price"] or 'N/A'

        # Extract fundamentals
        fundamentals = {
            "Company": info.get("longName") or info.get("shortName") or "N/A",
            "Business Summary":info.get('longBusinessSummary') or "N/A",
            "Sector":info.get('sector','N/A'),
            "Industry":info.get('industry','N/A'),
            'Open': info.get('open','N/A'),
            'Day Low': info.get('dayLow','N/A'),
            'Day High': info.get('dayHigh','N/A'),
            "PE Ratio (trailing)": info.get("trailingPE",'N/A'),
            "Forward PE": info.get("forwardPE",'N/A'),
            "ROE (%)": f"{(info.get('returnOnEquity', 0) * 100):.2f}%" if info.get('returnOnEquity') else 'N/A',
            "Debt to Equity Ratio": info.get("debtToEquity",'N/A'),
            "Total Revenue (ttm)": info.get("totalRevenue",'N/A'),
            "Net Income (ttm)": info.get("netIncomeToCommon",'N/A'),
            "Revenue Growth (%)": f"{(info.get('revenueGrowth', 0) * 100):.2f}%" if info.get('revenueGrowth') else 'N/A',
            "Current Price": current_price,
            "Market Cap": info.get("marketCap",'N/A'),
            "Profit Margins": info.get("profitMargins",'N/A'),
            'Operating Margins': info.get('operatingMargins','N/A'),
            'Free Cashflow': info.get('freeCashflow','N/A'),
            'Operating Cashflow': info.get('operatingCashflow','N/A'),
            'FiftyTwo Week High': info.get('fiftyTwoWeekHigh','N/A'),
            'FiftyTwo Week Low': info.get('fiftyTwoWeekLow','N/A'),
        }

        return {'fundamental_data':fundamentals}

    except Exception as e:
        fallback_data = {"Error": "No data available"}
        return {'fundamental_data':fallback_data}



# find recent business news
def recent_business_news(state:StockState):
    try:
        gn = GoogleNews(country='IN')
        business = gn.topic_headlines('business')

        summary=[]
        for entry in business.get('entries', []):
            for sub_article in entry.get('sub_articles', []):
                summary.append(sub_article.get('title'))
    except:
        return {'final_result':'Error in fetching the recent news. Please Try Again'}

    batch_size=35
    business_summary=""
    for i in range(0,len(summary),batch_size):
        prompt=f""" 
            You are a professional business data reporter.

            You will be given a text of news articles collected from multiple sources.

            Instructions:
            * Identify and REMOVE duplicate or near-duplicate news (even if the wording or source is different).
            * FILTER OUT news that is NOT related to business, finance, markets, companies, economy, or investments.
            * GROUP related news together (e.g., multiple articles about the same company or event).
            * Produce a CLEAR and CONCISE summary of the remaining business news in 10-20 lines .

            Just give me summarise output in 4-5 lines and nothing else. Don't give any heading to the out put. Give output in pointer form with mark '*'. 
            ### Input News:
            {"".join(summary[i:i+batch_size])}

            """

        result=llm.invoke(prompt)
        business_summary=business_summary+result.content

    return {'final_result':business_summary}


def Recent_stock_News(state:StockState):
    try:
        summary=[]
        gn = GoogleNews(country='IN')
        search = gn.search(state.get('stock_name'))
        for sub_search in search.get('entries',[]):
            summary.append(sub_search.get('title'))
    except:
        return {'latest_stock_news':'No Data available or Error in fetching news'}

    batch_size=35
    business_summary=""
    for i in range(0,len(summary),batch_size):
        prompt=f""" 
            You are a professional business data reporter.

            You will be given a text of news articles collected from multiple sources.

            Instructions:
            1. Identify and REMOVE duplicate or near-duplicate news (even if the wording or source is different).
            2. FILTER OUT news that is NOT related to business, finance, markets, companies, economy, or investments.
            3. GROUP related news together (e.g., multiple articles about the same company or event).
            4. Produce a CLEAR and CONCISE summary of the remaining business news in 5-10 lines .

            Just give me summarise output in paragaph format and nothing else.
            ### Input News:
            {"".join(summary[i:i+batch_size])}

            """

        result=llm.invoke(prompt)
        business_summary=business_summary+result.content
    
    return {'latest_stock_news':business_summary}


# This will find recent news about that particular stock
def Recent_stock_News_only(state:StockState):
    try:
        summary=[]
        gn = GoogleNews(country='IN')
        search = gn.search(state.get('stock_name'))
        for sub_search in search.get('entries',[]):
            summary.append(sub_search.get('title'))
    except:
        return {'latest_stock_news':'No Data available or Error in fetching news'}


    batch_size=35
    business_summary=""
    for i in range(0,len(summary),batch_size):
        prompt=f""" 
            You are a professional business data reporter.

            You will be given a text of news articles collected from multiple sources.

            Instructions:
            1. Identify and REMOVE duplicate or near-duplicate news (even if the wording or source is different).
            2. FILTER OUT news that is NOT related to business, finance, markets, companies, economy, or investments.
            3. GROUP related news together (e.g., multiple articles about the same company or event).
            4. Produce a CLEAR and CONCISE summary of the remaining business news in 5-10 lines .

            Just give me summarise output in paragraph format and nothing else.
            ### Input News:
            {"".join(summary[i:i+batch_size])}

            """

        result=llm.invoke(prompt)
        business_summary=business_summary+result.content
    
    return {'final_result':business_summary}


# This will combine stock news and fundamentals 
def Buy_sell_analyst(state:StockState):

    prompt = ChatPromptTemplate.from_messages([
        ("system", """ You are a professional stock market equity research analyst.
            Your job is to analyze a stock objectively using:
            - Fundamental financial data
            - Latest company-specific news
            - Broader business and macroeconomic news

            You must think like a real analyst:
            - Weigh positives vs negatives
            - Avoid hype or emotional bias
            - Clearly justify every conclusion

            Your final output must be structured, concise, and actionable."""),

        ("human", """
                User Query:{user_query},
                Latest Stock-Specific News:{latest_stock_news},
                Fundamental Data of the Stock:{fundamental_data},
                Analyze the stock as a professional investment analyst and provide an investment recommendation.
         """)])

    chain=prompt|llm
    results=chain.invoke({'user_query':state['user_query'],
                          'latest_stock_news':state['latest_stock_news'],
                          'fundamental_data':state['fundamental_data']})
    
    return {'final_result':results.content}
    

# for conditional edges 
# One conditon after intent node 
def check_intent(state:StockState)->Literal['Stock_name','General_query_handler','recent_business_news']:
    if state['intent']=='buy sell':
        return 'Stock_name'
    if state['intent']=='general talk':
        return 'General_query_handler'
    if state['intent']=='business news':
        return 'recent_business_news' 
    else:
        return 'Stock_name'
        
# one condition at Stock name. Two path - 1. For only stock news 2. For buy sell analysis 
def route_after_stock_name(state: StockState) -> Literal['Ticker_finder', 'Recent_stock_News_only']:
    intent = state.get('intent', '')
    if intent=='stock news':
        return 'Recent_stock_News_only'
    else:
        return 'Ticker_finder'
    


graph=StateGraph(StockState)

graph.add_node('Intent_finder',Intent_finder)
graph.add_node('General_query_handler',General_query_handler)
graph.add_node('Stock_name',Stock_name_finder)
graph.add_node('recent_business_news',recent_business_news)
graph.add_node('Ticker_finder',Ticker_finder)
graph.add_node('Fundamental_finder',Fundamental_finder)
graph.add_node('Recent_stock_News',Recent_stock_News)
graph.add_node('Buy_sell_analyst',Buy_sell_analyst)
graph.add_node('Recent_stock_News_only',Recent_stock_News_only)


graph.add_edge(START,'Intent_finder')

graph.add_conditional_edges('Intent_finder',
                            check_intent,
                            {
                            'Stock_name':'Stock_name',
                            'General_query_handler':'General_query_handler',
                            'recent_business_news':'recent_business_news',
                            })

# general query handler done
graph.add_edge('General_query_handler',END)
#only business news
graph.add_edge('recent_business_news',END)

# stock name to ticker and recent stock
graph.add_conditional_edges('Stock_name',
                            route_after_stock_name,
{
    'Recent_stock_News_only':'Recent_stock_News_only',
    'Ticker_finder':'Ticker_finder'
})



# handle stock news only section
graph.add_edge('Recent_stock_News_only',END)


graph.add_edge('Ticker_finder','Fundamental_finder')
graph.add_edge('Ticker_finder','Recent_stock_News')
graph.add_edge('Fundamental_finder','Buy_sell_analyst')
graph.add_edge('Recent_stock_News','Buy_sell_analyst')
graph.add_edge('Buy_sell_analyst',END)


checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer) 



# CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# # session state for message history
# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []


# for message in st.session_state['message_history']:
#     with st.chat_message(message['role']):
#         st.text(message['content'])

# query = st.chat_input("Ask about a stock or market:")

# if query:
#     st.session_state['message_history'].append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.text(query)


#     response=workflow.invoke({'user_query':query},config=CONFIG)
#     st.session_state['message_history'].append({"role": "assistant", "content": response['final_result']})
#     with st.chat_message("assistant"):
#         st.markdown(response['final_result'])
