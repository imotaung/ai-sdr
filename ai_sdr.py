#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
APOLLO_API_KEY = os.environ["APOLLO_API_KEY"]
RESEND_API_KEY = os.environ["RESEND_API_KEY"]
FROM_EMAIL = os.environ["FROM_EMAIL"]
NOTIFICATION_EMAIL = os.environ["NOTIFICATION_EMAIL"]
GOOGLE_SHEET_ID = os.environ["GOOGLE_SHEET_ID"]
GOOGLE_CREDENTIALS_JSON = os.environ["GOOGLE_CREDENTIALS_JSON"]
ICP_FILTERS = json.loads(os.environ["ICP_FILTERS"])

LLM_MODEL = os.environ.get("LLM_MODEL", "groq/llama3-70b-8192")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

MAX_LEADS = 200
RESEND_RATE = 2

# Apollo endpoint (company search only – people search requires paid plan)
APOLLO_COMPANY_URL = "https://api.apollo.io/v1/organizations/search"

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def apollo_request(method: str, url: str, json_data: Dict = None) -> Dict:
    headers = {"Content-Type": "application/json", "X-API-Key": APOLLO_API_KEY}
    if method.lower() == "get":
        resp = requests.get(url, headers=headers, params=json_data, timeout=30)
    else:
        resp = requests.post(url, headers=headers, json=json_data, timeout=30)
    resp.raise_for_status()
    return resp.json()

def find_companies(icp: Dict, limit: int = 100) -> List[Dict]:
    companies = []
    page = 1
    per_page = 25
    while len(companies) < limit:
        filters = {}
        locations = icp.get("locations", [])
        if locations:
            filters["organization_locations"] = locations
        
        min_emp = icp.get("min_employees", 10)
        emp_range = _employee_range(min_emp)
        if emp_range:
            filters["num_employees_ranges"] = [emp_range]
        
        payload = {
            "page": page,
            "per_page": per_page,
            "filters": filters
        }
        if not filters:
            payload.pop("filters", None)
        
        logger.debug(f"Payload: {json.dumps(payload)}")
        
        data = apollo_request("post", APOLLO_COMPANY_URL, json_data=payload)
        batch = data.get("organizations", [])
        if not batch:
            break
        companies.extend(batch)
        
        pagination = data.get("pagination", {})
        total_entries = pagination.get("total_entries", 0)
        if total_entries and total_entries <= len(companies):
            break
        page += 1
        time.sleep(0.5)
    return companies[:limit]

def _employee_range(min_emp: int) -> str:
    if min_emp <= 10:
        return "1-10"
    elif min_emp <= 50:
        return "11-50"
    elif min_emp <= 200:
        return "51-200"
    elif min_emp <= 500:
        return "201-500"
    elif min_emp <= 1000:
        return "501-1000"
    else:
        return "1000+"

def get_fallback_email(domain: str) -> str:
    """Return a generic email address for the domain (since Apollo people search is paid)."""
    # Common generic prefixes
    prefixes = ["hello", "contact", "info", "hi", "team"]
    # You can rotate or try multiple; for simplicity we use "hello"
    return f"hello@{domain}"

def generate_email(company: Dict, recipient_email: str) -> str:
    name = company.get("name", "the company")
    industry = company.get("industry", "your industry")
    location = company.get("headquarters", {}).get("city", company.get("location", "your area"))
    description = company.get("short_description", f"a company in {industry}")

    prompt = f"""Write a short, friendly B2B sales email to {recipient_email} about {name}.
Company: {name}
Industry: {industry}
Location: {location}
Description: {description}

Rules:
- Reference something specific about the company (industry or location).
- Max 150 words, warm tone.
- End with: "Would you be open to a quick 15-min chat? Reply with '?' if interested."
- No subject line.

Email body:"""
    try:
        response = completion(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                              temperature=0.7, max_tokens=300)
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return f"""Hi there,

I've been following {name} in {industry}. I'd love to share a few growth ideas.

Would you be open to a quick 15-min chat? Reply with '?' if interested.

Best, AI SDR"""

def send_email(to: str, subject: str, body: str) -> bool:
    time.sleep(1.0 / RESEND_RATE)
    url = "https://api.resend.com/emails"
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"}
    payload = {"from": FROM_EMAIL, "to": [to], "subject": subject, "text": body}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info(f"Sent to {to}")
        return True
    except Exception as e:
        logger.error(f"Send failed to {to}: {e}")
        return False

def init_sheet():
    creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(GOOGLE_SHEET_ID).sheet1
    headers = ["timestamp", "company_name", "domain", "decision_maker_email", "generated_email_content", "status"]
    if not sheet.row_values(1):
        sheet.insert_row(headers, 1)
    return sheet

def log_lead(sheet, company: str, domain: str, email: str, content: str, status: str):
    # Use timezone-aware UTC datetime to avoid deprecation warning
    timestamp = datetime.now(timezone.utc).isoformat()
    sheet.append_row([timestamp, company, domain, email, content, status])

def notify_admin(subject: str, body: str):
    send_email(NOTIFICATION_EMAIL, subject, body)

def main():
    try:
        logger.info("AI SDR started")
        sheet = init_sheet()
        companies = find_companies(ICP_FILTERS, limit=MAX_LEADS)
        logger.info(f"Found {len(companies)} companies")

        stats = {"found": len(companies), "enriched": 0, "sent": 0, "failed": 0}

        for company in companies:
            name = company.get("name", "Unknown")
            domain = company.get("primary_domain", "")
            if not domain:
                log_lead(sheet, name, "", "", "", "skipped (no domain)")
                continue

            # Use fallback generic email instead of Apollo people search
            decision_email = get_fallback_email(domain)
            logger.info(f"Using fallback email {decision_email} for {name}")

            stats["enriched"] += 1
            email_body = generate_email(company, decision_email)
            subject = f"Quick question about {name}"
            success = send_email(decision_email, subject, email_body)
            status = "sent" if success else "failed (send error)"
            if success:
                stats["sent"] += 1
            else:
                stats["failed"] += 1
            log_lead(sheet, name, domain, decision_email, email_body, status)

            if stats["enriched"] >= 200:
                logger.info("Reached daily limit, stopping")
                break

        summary = f"AI SDR completed\nFound: {stats['found']}\nEnriched: {stats['enriched']}\nSent: {stats['sent']}\nFailed: {stats['failed']}"
        notify_admin("AI SDR Success", summary)
        logger.info(summary)

    except Exception as e:
        error_msg = f"AI SDR CRASH:\n{str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        notify_admin("AI SDR Failed", error_msg)
        raise

if __name__ == "__main__":
    main()