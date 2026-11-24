"""Generate sample datasets: 3 policy PDFs (40/10/15 pages) + past_hr_tickets.csv (1000 rows).
Run:  python data_generator.py
"""
import random
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from config import DATA_DIR

HANDBOOK = [
    ("Attendance and Working Hours", [
        "Standard working hours are 9:00 AM to 6:00 PM, Monday to Friday, including a one-hour lunch break.",
        "A grace period of 15 minutes is allowed up to three times per month; beyond that a half-day leave is deducted."],
     ["Employees mark attendance daily through the HR portal.", "Managers review attendance exceptions weekly."],
     [("What are the office hours?", "Office hours are 9:00 AM to 6:00 PM, Monday to Friday."),
      ("How many late arrivals are tolerated?", "Three grace-period arrivals of up to 15 minutes per month.")]),
    ("Work From Home", [
        "Employees may work from home up to 2 days per week with prior manager approval.",
        "During WFH, employees must be available during core hours of 11:00 AM to 4:00 PM."],
     ["Raise a WFH request in the HR portal one day in advance.", "Managers approve based on team needs."],
     [("How many WFH days do I get?", "Up to 2 days per week with manager approval."),
      ("Are there core hours on WFH days?", "Yes, 11:00 AM to 4:00 PM.")]),
    ("Reimbursements", [
        "Travel claims must be submitted within 30 days of the trip along with original receipts.",
        "A WFH internet allowance of Rs 1,000 per month is reimbursed against a bill.",
        "Meal allowance during official travel is Rs 500 per day."],
     ["Submit claims in the finance portal with scanned receipts.", "Finance processes approved claims within 10 working days."],
     [("What is the deadline for travel claims?", "30 days from the trip."),
      ("How much is the internet allowance?", "Rs 1,000 per month against a bill.")]),
    ("Dress Code", [
        "The dress code is business casual from Monday to Thursday and casual on Fridays.",
        "Client meetings require formal attire."],
     ["Managers may remind employees of the dress code privately.", "Repeated violations are handled by HR."],
     [("Can I wear jeans?", "Casual wear, including jeans, is allowed on Fridays only.")]),
    ("Probation and Notice Period", [
        "The probation period for new employees is 6 months.",
        "The notice period for confirmed employees is 60 days."],
     ["Confirmation is communicated in writing by HR after a probation review.", "Notice period buy-out requires HR and management approval."],
     [("How long is probation?", "6 months."), ("What is the notice period?", "60 days for confirmed employees.")]),
    ("Performance Reviews", [
        "Performance reviews are conducted twice a year, in April and October.",
        "Ratings are given on a 5 point scale."],
     ["Employees complete a self-assessment before each review.", "Managers hold a one-to-one feedback discussion."],
     [("When are reviews held?", "In April and October every year.")]),
    ("Holidays", [
        "The company observes 10 public holidays per year, plus 2 optional holidays chosen from a published list.",
        "The holiday calendar is published in December for the next year."],
     ["Optional holidays must be applied for through the HR portal.", "Holidays falling on weekends are not compensated."],
     [("How many public holidays are there?", "10 public holidays plus 2 optional holidays.")]),
    ("IT and Information Security", [
        "Passwords must be changed every 90 days.",
        "Lost or stolen company devices must be reported to IT within 24 hours."],
     ["Use only company-approved software on laptops.", "Report phishing emails to the security desk."],
     [("How often should I change my password?", "Every 90 days."),
      ("What if I lose my laptop?", "Report it to IT within 24 hours.")]),
    ("Health Benefits", [
        "Every employee is covered by a group health insurance of Rs 5 lakh on a family floater basis.",
        "An annual health check-up is provided free of cost."],
     ["Claims are cashless at network hospitals.", "Add dependants within 30 days of joining or life events."],
     [("What is the health insurance cover?", "Rs 5 lakh family floater.")]),
    ("Learning and Development", [
        "Each employee has an annual learning budget of Rs 25,000 for courses and certifications.",
        "Approval from the manager is required before enrollment."],
     ["Submit course details and cost to your manager.", "Reimbursement is made after course completion proof."],
     [("What is the learning budget?", "Rs 25,000 per year per employee.")]),
]

LEAVE = [
    ("Casual Leave", [
        "Employees receive 12 days of casual leave per calendar year.",
        "A maximum of 3 consecutive casual leave days may be taken at a time.",
        "Unused casual leave cannot be carried forward and lapses on 31 December."],
     ["Apply through the HR portal.", "Casual leave cannot be clubbed with earned leave beyond 5 days."], []),
    ("Sick Leave", [
        "Employees receive 10 days of sick leave per year.",
        "A medical certificate is mandatory for sick leave exceeding 2 consecutive days.",
        "Up to 5 unused sick leave days can be carried forward."],
     ["Inform your manager on the first day of illness.", "Upload the medical certificate on return."], []),
    ("Earned Leave", [
        "Earned leave accrues at 1.5 days per month, i.e. 18 days per year.",
        "Up to 30 days of earned leave can be carried forward.",
        "Earned leave can be encashed at the time of resignation or retirement."],
     ["Plan earned leave at least a week in advance.", "Encashment is processed with the final settlement."], []),
    ("Maternity and Paternity Leave", [
        "Female employees are entitled to 26 weeks of paid maternity leave.",
        "Male employees are entitled to 10 working days of paternity leave."],
     ["Submit the request with a medical document to HR.", "HR confirms dates within 3 working days."], []),
    ("Leave Application", [
        "Planned leave must be applied through the HR portal at least 3 working days in advance.",
        "Managers must approve or reject within 2 working days.",
        "Emergency leave may be applied retrospectively within 2 days of return."],
     ["Leave balance is visible on the HR portal.", "Unapproved absence is treated as loss of pay."], []),
]

CONDUCT = [
    ("Ethics and Integrity", [
        "Employees must act honestly and avoid any form of bribery or corruption.",
        "Violations may lead to disciplinary action up to termination."],
     ["Report suspected violations to your manager or HR."], [("What happens on an ethics violation?", "Disciplinary action up to termination.")]),
    ("Anti-Harassment and POSH", [
        "The company has zero tolerance for sexual harassment under the POSH Act.",
        "Complaints can be filed with the Internal Committee within 3 months of the incident.",
        "All complaints are handled confidentially."],
     ["Contact the Internal Committee directly for any complaint."], [("Where do I report harassment?", "To the Internal Committee, within 3 months.")]),
    ("Conflict of Interest", [
        "Gifts worth more than Rs 2,000 from vendors or clients must be declared to HR.",
        "Employees must not hold outside employment without written approval."],
     ["Declare conflicts in the annual disclosure form."], [("What gifts must I declare?", "Gifts worth more than Rs 2,000.")]),
    ("Confidentiality", [
        "Confidential company and client information must not be shared outside the organisation.",
        "The confidentiality obligation continues after employment ends."],
     ["Sign the NDA at joining.", "Use only approved channels to share data."], [("Does confidentiality end when I leave?", "No, it continues after employment ends.")]),
    ("Whistleblower and Social Media", [
        "Concerns can be reported anonymously to ethics@company.example.",
        "Employees must not post confidential information or represent personal views as the company's on social media."],
     ["Anonymous reports are reviewed by the ethics committee."], [("How do I report anonymously?", "Email ethics@company.example.")]),
]

PARTS = {
    "Purpose and Scope": lambda t, f, p, q: [
        f"This section explains the company policy on {t.lower()}. It applies to all full-time employees, "
        "interns and contract staff unless stated otherwise.",
        "The policy aims to ensure fairness, transparency and legal compliance across the organisation."],
    "Policy Details": lambda t, f, p, q: f + [
        "In case of ambiguity, the interpretation of the HR department shall be final."],
    "Procedure and Responsibilities": lambda t, f, p, q: p + [
        "Employees are responsible for following the procedure; managers are responsible for timely approvals; "
        "HR is responsible for record keeping."],
    "Frequently Asked Questions": lambda t, f, p, q: (
        [f"Q: {a}  A: {b}" for a, b in q] or [f"Q: Where can I read more about {t.lower()}?  A: Refer to the policy details page."]),
}


def _footer(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.drawCentredString(A4[0] / 2, 20, f"Page {doc.page}")


def build_pdf(fname, title, topics, parts):
    styles = getSampleStyleSheet()
    story, n = [], 0
    for ti, (t, f, p, q) in enumerate(topics, 1):
        for pi, part in enumerate(parts, 1):
            n += 1
            story.append(Paragraph(f"Section {ti}.{pi}: {t} - {part}", styles["Heading1"]))
            story.append(Spacer(1, 12))
            for para in PARTS[part](t, f, p, q):
                story.append(Paragraph(para, styles["BodyText"]))
                story.append(Spacer(1, 8))
            story.append(PageBreak())
    SimpleDocTemplate(str(DATA_DIR / fname), pagesize=A4, title=title).build(
        story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"{fname}: {n} pages")


def build_tickets(n=1000):
    random.seed(42)
    templates = [
        ("Leave", "How many casual leaves do I get?", "12 days per year."),
        ("Leave", "Can I carry forward my sick leave?", "Up to 5 days."),
        ("Leave", "How do I apply for earned leave?", "Through the HR portal, 3 days in advance."),
        ("Leave", "What is the maternity leave duration?", "26 weeks."),
        ("WFH", "How many WFH days are allowed?", "2 days per week."),
        ("Reimbursement", "What is the deadline for travel claim?", "30 days."),
        ("Reimbursement", "Is internet bill reimbursed?", "Rs 1,000 per month."),
        ("Attendance", "What are office timings?", "9 AM to 6 PM."),
        ("Benefits", "What is my health insurance cover?", "Rs 5 lakh family floater."),
        ("Benefits", "Is there a learning budget?", "Rs 25,000 per year."),
        ("Payroll", "When will salary be credited?", "Last working day of the month."),
        ("IT", "How often do I change my password?", "Every 90 days."),
        ("Conduct", "Do I need to declare gifts?", "Yes, above Rs 2,000."),
        ("Exit", "What is the notice period?", "60 days."),
        ("Holidays", "How many optional holidays?", "2 optional holidays."),
    ]
    rows = []
    for i in range(1, n + 1):
        c, q, r = random.choice(templates)
        rows.append({"ticket_id": f"HR{i:05d}", "category": c, "query": q, "resolution": r,
                     "status": random.choice(["Resolved", "Resolved", "Resolved", "Escalated"]),
                     "created_date": f"2026-{random.randint(1, 10):02d}-{random.randint(1, 28):02d}"})
    pd.DataFrame(rows).to_csv(DATA_DIR / "past_hr_tickets.csv", index=False)
    print(f"past_hr_tickets.csv: {n} rows")


if __name__ == "__main__":
    build_pdf("employee_handbook.pdf", "Employee Handbook", HANDBOOK,
              ["Purpose and Scope", "Policy Details", "Procedure and Responsibilities", "Frequently Asked Questions"])
    build_pdf("leave_policy.pdf", "Leave Policy", LEAVE, ["Policy Details", "Procedure and Responsibilities"])
    build_pdf("code_of_conduct.pdf", "Code of Conduct", CONDUCT,
              ["Purpose and Scope", "Policy Details", "Frequently Asked Questions"])
    build_tickets()
