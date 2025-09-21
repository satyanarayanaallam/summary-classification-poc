import json
import random
from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / 'data' / 'summaries.json'

# Templates per doc type for longer, varied summaries
TEMPLATES = {
    'INVOICE': [
        "Invoice {code} records a transaction between the supplier and the purchaser for goods or services provided during the billing period, listing quantities, unit prices, taxes, total amount due, and payment terms. The document indicates the invoice date, due date, and a reference to the related purchase order.",
        "This invoice ({code}) itemizes charges for provided services with clear line-item descriptions, subtotal, applicable taxes, shipping fees, and the grand total. It specifies payment instructions and remittance details to facilitate timely settlement.",
        "A detailed billing notice identified as {code}, summarizing the products delivered, service durations, unit rates, discounts applied, and the final payable amount along with bank transfer details and contact information for billing queries."
    ],
    'BANK_STATEMENT': [
        "Bank statement {code} summarizes account activity over the statement period, including opening and closing balances, itemized deposits and withdrawals, interest earned, service charges, and notes on any pending transactions to reconcile the account.",
        "This monthly account statement ({code}) provides a chronological listing of transactions, date-wise debits and credits, checks cleared, ATM withdrawals, and end-of-period balances, helping account holders track cash flows and detect discrepancies.",
        "A comprehensive financial statement for account {code} detailing received payments, transfers, automated debits, fees applied, and balance adjustments, with a section highlighting any overdrafts or returned items requiring attention."
    ],
    'LEAVE_REQUEST': [
        "Leave request {code} documents an employee's formal application for time off, specifying requested dates, total leave duration, reason for absence, contact details during leave, and manager approval status or required coverage arrangements.",
        "This request ({code}) outlines the type of leave requested (annual, sick, unpaid), start and end dates, work handover notes for colleagues, and any supporting information such as medical certificates or travel plans to justify the time away.",
        "An employee leave submission labeled {code} describing the anticipated absence timeframe, the employee's rationale, proposed interim point-of-contact for ongoing tasks, and an approval trail for HR recordkeeping."
    ],
    # Placeholder templates for potential additional doc types
}

# Generic filler sentences to add variation
EXTRAS = [
    "The document includes contact information and identifies responsible parties for any clarification or dispute resolution.",
    "It highlights key dates and monetary figures so recipients can quickly see responsibilities and deadlines.",
    "The summary emphasizes compliance-related notes where applicable, such as tax implications or contractual obligations.",
    "Records any reference numbers or authorization codes required for internal tracking and audit.",
    "Notes any pending actions or follow-ups that the recipient must complete within the stated timeframe."
]


def load_data(path: Path):
    text = path.read_text(encoding='utf-8')
    # Strip Markdown fences if present
    if text.lstrip().startswith('```'):
        # Remove first fence line and last fence line
        lines = text.splitlines()
        # find opening fence index
        start = 0
        while start < len(lines) and lines[start].strip().startswith('```'):
            start += 1
        # find closing fence index
        end = len(lines) - 1
        while end >= 0 and lines[end].strip().startswith('```'):
            end -= 1
        content = '\n'.join(lines[start:end+1])
    else:
        content = text
    return json.loads(content)


def save_data(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding='utf-8')


def expand_summaries(data):
    for rec in data:
        doc_type = rec.get('doc_type')
        code = rec.get('doc_code')
        templates = TEMPLATES.get(doc_type, None)
        if templates:
            # pick 1-2 templates and 1-2 extras to form a longer varied summary
            parts = []
            parts.append(random.choice(templates).format(code=code))
            # sometimes add another template sentence
            if random.random() < 0.5:
                parts.append(random.choice(templates).format(code=code))
            parts.extend(random.sample(EXTRAS, k=random.randint(1, 2)))
            rec['summary'] = ' '.join(parts)
        else:
            # fallback longer description
            rec['summary'] = f"{doc_type} record {code} with transaction details and administrative notes."
    return data


if __name__ == '__main__':
    data = load_data(DATA_PATH)
    print(f"Loaded {len(data)} records")
    data = expand_summaries(data)
    save_data(DATA_PATH, data)
    print(f"Updated and saved {len(data)} records to {DATA_PATH}")
