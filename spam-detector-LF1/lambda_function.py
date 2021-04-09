import json
import boto3
import email
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences
import os
import io
from botocore.exceptions import ClientError

# Machine learning notebook endpoint
ENDPOINT_NAME = "sms-spam-classifier-mxnet-2021-04-06-03-17-06-255"
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

def lambda_handler(event, context):
    s3 = boto3.client('s3', region_name='us-east-1')
    info = event['Records'][0]['s3']
    bucket = info['bucket']['name']
    name = info['object']['key']
    
    # Get the email
    file = s3.get_object(Bucket=bucket, Key=name)['Body'].read()
    
    message = email.message_from_string(file.decode("utf-8"))
    #print(message)
    ret_body = ''
    if message.is_multipart():
        for pl in message.get_payload():
            ret_body += pl.get_payload()
    else:
        ret_body += message.get_payload()
    
    print(ret_body)
            
    ret_body = ret_body.replace("=E2=80=93=20\r\n", "\r\n")
    ret_body = ret_body.replace("=\r\n", "\r\n").replace("\r\n=20", "\r\n")
    print(ret_body)
    body = ret_body.replace("\n","").replace("\r", "")
    print(body)
    
    # Sagemaker part
    runtime= boto3.client('runtime.sagemaker')
    # Prepare the message to test
    vocabulary_length = 9013
    test_msg = [body]
    one_hot_test_messages = one_hot_encode(test_msg, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    encoded_json_msg = json.dumps(encoded_test_messages.tolist())
    
    runtime= boto3.client('runtime.sagemaker', region_name='us-east-1')
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                        ContentType='text/csv',
                                        Body=encoded_json_msg)
    response = json.loads(response['Body'].read().decode())
    print("Response: ", response)
    score = int(response["predicted_label"][0][0])
    Classification = 'HAM' if score == 0 else "SPAM"
    Probability = response["predicted_probability"][0][0]
    
    # Prepare the return email
    aws_region = "us-east-1"
    sender = message['To']
    recipient = message['From']
    subject = "Analysis for your email"
    body_text = "We received your email sent at {} with the subject {}.\r\n".format(message['Date'], message['SUBJECT'])
    body_text += "\r\nHere is a 240 character sample of the email body:\r\n"
    body_text += "\r\n" + ret_body[:240] + "\r\n"
    body_text += "\r\nThe email was categorized as {} ".format(Classification)
    body_text += "with a {:.5%} confidence.".format(Probability)
    
    send_email(sender, recipient, aws_region, subject, body_text)
    
    return {
        'statusCode': 200,
        'body': json.dumps('We have already know if your email is spam or not')
    }
    
    
    
    
    
def send_email(sender, recipient, aws_region, subject, body_text):
    CHARSET = "UTF-8"
    client = boto3.client('ses',region_name = aws_region)
    
    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination = {
                'ToAddresses': [
                    recipient,
                ],
            },
            Message = {
                'Body': {
                    'Text': {
                        'Charset': CHARSET,
                        'Data': body_text,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': subject,
                },
            },
            Source=sender,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Sent!, the ID is:", response['MessageId'])


