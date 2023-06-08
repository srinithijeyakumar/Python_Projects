#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client')


# In[ ]:


import time
import random
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API Scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Gmail ID and Vacation Response Message
GMAIL_ID = 'xxxxx'
VACATION_RESPONSE = 'Thank you for your email. I am currently on vacation and will respond to your message when I return.'

# Function to get Gmail API service
def get_gmail_service():
    creds = None
    flow = InstalledAppFlow.from_client_secrets_file(R'xxx', SCOPES)
    creds = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=creds)

# Function to check for new emails
def check_emails(service):
    try:
        response = service.users().messages().list(userId=GMAIL_ID, q='is:unread').execute()
        messages = response.get('messages', [])

        for message in messages:
            msg = service.users().messages().get(userId=GMAIL_ID, id=message['id']).execute()
            headers = msg['payload']['headers']
            subject = [header['value'] for header in headers if header['name'] == 'Subject'][0]
            
            if not has_prior_reply(msg):
                reply_to_email(service, msg)
                mark_email_as_replied(service, msg)
                print(f"Replied to email: {subject}")

    except HttpError as error:
        print(f"An error occurred: {error}")

# Function to check if an email has a prior reply
def has_prior_reply(email):
    for header in email['payload']['headers']:
        if header['name'] == 'In-Reply-To' or header['name'] == 'References':
            return True
    return False

# Function to send a reply to an email
def reply_to_email(service, email):
    reply = {
        'raw': base64.urlsafe_b64encode(
            f"From: {GMAIL_ID}\nTo: {email['payload']['headers'][18]['value']}\nSubject: {email['payload']['headers'][8]['value']}\n\n{VACATION_RESPONSE}".encode("utf-8")
        ).decode("utf-8")
    }

    service.users().messages().send(userId=GMAIL_ID, body=reply).execute()

# Function to mark an email as replied and move it to a label
def mark_email_as_replied(service, email):
    msg_id = email['id']
    thread_id = email['threadId']

    service.users().messages().modify(userId=GMAIL_ID, id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
    service.users().messages().modify(userId=GMAIL_ID, id=msg_id, body={'addLabelIds': ['REPLIED']}).execute()
    service.users().threads().modify(userId=GMAIL_ID, id=thread_id, body={'addLabelIds': ['REPLIED']}).execute()

# Main loop
def main():
    service = get_gmail_service()

    while True:
        check_emails(service)
        interval = random.randint(45, 120)
        time.sleep(interval)

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




