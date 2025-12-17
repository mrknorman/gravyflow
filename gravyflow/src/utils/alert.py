import smtplib
import logging

logger = logging.getLogger(__name__)
from pathlib import Path
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email(
        subject : str, 
        message : str, 
        to_email : str, 
        config_path : Path
    ):

    # Load email details from JSON file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    from_email = config['your_email']
    password = config['your_password']  # or your App Password if 2FA is enabled

    # Set up the MIME
    mime = MIMEMultipart()
    mime['From'] = from_email
    mime['To'] = to_email
    mime['Subject'] = subject

    # Attach the message
    mime.attach(MIMEText(message, 'plain'))

    # Create SMTP session for sending the mail
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # use Gmail with port
        server.starttls()  # enable security
        server.login(from_email, password)  # login with mail_id and password
        text = mime.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        logger.info(f"Sent email to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")