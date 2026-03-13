import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

import a_passwards as pw


def send_html_email_with_attachment(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    password: str,
    receiver_email: str,
    subject: str,
    html_body: str,
    attachment_path: str | None = None,
):
    # Root message (mixed)
    msg = MIMEMultipart("mixed")
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # Alternative part (plain + html)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText("This email contains an HTML report.", "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    # Attachment
    if attachment_path:
        path = Path(attachment_path)
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{path.name}"'
        )
        msg.attach(part)

    # Send
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)

    print("Email sent successfully.")


if __name__ == '__main__':
    HTML_PATH = "back_test/daily_results/all_results_combine.html"  # your report path
    try:
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            HTML_BODY = f.read()
    except Exception:
        HTML_BODY = "<p>Please find the attached file.</p>"

    
    for re in pw.RECIPIENTS:
        send_html_email_with_attachment(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email=pw.SENDER_EMAIL,    # your gmail
            password=pw.google_email_app_password,  # your gmail app password
            receiver_email=re,  # recipient email
            subject="Daily Backtest Report",
            html_body=HTML_BODY,
            attachment_path=HTML_PATH
        )
