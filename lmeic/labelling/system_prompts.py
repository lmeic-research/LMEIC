#!/usr/bin/env python3
"""
Phase 1 Annotation System Prompts: 4-Agent + Adjudicator + Debate templates.

Version: v2.0 | Date: 2026-05-27

Revision from v1.0:
- Recalibrated Agent D (risk-orientation): added proportionality constraint
  to prevent over-escalation to High (was 45.0% High, target ~30%)
- Recalibrated Agent B (action-urgency): broadened urgency scope beyond
  explicit deadlines only (was 28.6% Low, target ~17%)
- Added shared CALIBRATION_ANCHORS with 2 examples per class per role
- Sharpened H/M/L boundary definitions in ROLE_CRITERIA
- Added DECISION_BOUNDARY_RULES shared across all agents
- Cross-level conflicts (H<->L) target: <=3% (was 5.4%)

Root cause from v1 pilot (n=500, Fleiss kappa=0.408):
  Agent D voted High in 23/27 H<->L conflicts (risk over-escalation)
  Agent B voted Low in 23/27 H<->L conflicts (urgency under-detection)
  A-C pairwise agreement 76.6% (well-calibrated baseline)
  B-D pairwise agreement 54.0% (antipodal pulling, primary kappa drag)
"""

# =============================================================================
# SHARED DECISION BOUNDARY RULES (injected into ALL agent prompts)
# =============================================================================

DECISION_BOUNDARY_RULES = """
DECISION BOUNDARIES (apply these strictly before assigning a label):

HIGH requires AT LEAST ONE of:
  (a) An explicit same-day or next-business-day deadline/action requirement
  (b) Direct financial consequence if delayed (revenue loss, penalty, missed payment)
  (c) A named client/partner expressing dissatisfaction, escalation, or complaint
  (d) Compliance/legal/regulatory urgency with documented deadline
  (e) Confirmed operational disruption already in progress (not hypothetical)

MEDIUM requires AT LEAST ONE of:
  (a) A clear action item with a within-week time horizon (no same-day pressure)
  (b) Ongoing business matter that needs follow-up but has no immediate consequence
  (c) A request or question directed specifically to the recipient requiring a response
  (d) Operational coordination that affects business progress if ignored this week
  (e) Information that the recipient must read and potentially act upon (not just FYI)

LOW applies when NONE of the above conditions are met:
  - Mass notifications, newsletters, auto-generated reports
  - FYI-only forwards with no action expected from the recipient
  - Routine acknowledgments, thank-you notes, or confirmations with no follow-up
  - Content not relevant to the recipient's core role responsibilities
  - Duplicate or redundant information already received through other channels

CRITICAL CONSTRAINT:
  - Do NOT assign High based on HYPOTHETICAL or WORST-CASE scenarios alone.
    The risk must be CONCRETE and EVIDENT in the email content.
  - Do NOT assign Low merely because no explicit deadline is stated.
    Implicit urgency from context (ongoing negotiation, pending shipment, etc.) counts.
  - When genuinely uncertain between two adjacent levels, make your best judgment
    based on the available evidence. Do NOT systematically default to Medium —
    commit to the label that best matches the evidence.
"""

# =============================================================================
# ROLE CRITERIA (injected into all agent prompts based on email's role)
# =============================================================================

ROLE_CRITERIA = {
    "Mgr": """ROLE: Business Manager (Mgr)
Core responsibilities: Managing client/partner relationships, contracts, negotiations,
business development, and operational decision-making.

Classification criteria for this role:
- HIGH: Emails requiring same-day action — client escalations, contract deadlines,
  financial approvals, compliance alerts, partner disputes, or any content where delay
  causes DEMONSTRABLE (not hypothetical) business/revenue/relationship loss.
- MEDIUM: Emails requiring action within this week — routine follow-ups with clients,
  internal coordination, price/rate discussions, shipment status that needs attention,
  operational requests that affect business progress but are not time-critical today.
- LOW: Emails that can be batch-processed or archived — mass notifications, newsletters,
  FYI-only forwards, routine auto-generated reports, general announcements with no
  action required from the business manager.

CALIBRATION ANCHORS for Mgr:
  HIGH example: "Dear [name], our payment for invoice #2025-089 ($45,000) is overdue.
    If not received by Friday, we will suspend the service agreement per clause 7.3."
  MEDIUM example: "Hi [name], following up on last week's rate discussion. Could you
    confirm if you can match the competitor quote for the APAC lane? Thanks."
  LOW example: "Company newsletter: Q1 shipping industry trends and market outlook.
    Click here to read the full report." """,

    "CS": """ROLE: Overseas Customer Service (CS)
Core responsibilities: Handling customer inquiries, complaints, booking confirmations,
document processing, shipment tracking, and operational coordination with overseas agents.

Classification criteria for this role:
- HIGH: Emails requiring same-day response — customer complaints, urgent booking changes,
  document corrections with vessel deadline, missing pre-alerts for imminent sailings,
  customs clearance issues, or any content where delay causes DEMONSTRABLE service failure.
- MEDIUM: Emails requiring action within this week — routine booking confirmations,
  standard document requests, shipment status updates needing follow-up, rate inquiries,
  new customer onboarding tasks that affect service delivery but are not urgent today.
- LOW: Emails that can be batch-processed or archived — routine shipping schedules,
  mass rate circulars, internal newsletters, holiday notices, FYI-only forwards,
  or auto-generated tracking notifications with no action needed.

CALIBRATION ANCHORS for CS:
  HIGH example: "URGENT: Customer ABC Corp reports their cargo is stuck at customs due
    to wrong HS code on the BL. Vessel departs tomorrow. Please correct immediately."
  MEDIUM example: "Please prepare the shipping documents for booking BK20250315.
    The vessel ETD is next Wednesday. Standard set: BL + invoice + packing list."
  LOW example: "[Auto-notification] Vessel EVER GIVEN arrived at Shanghai port.
    All containers discharged. No action required." """,

    "Gen": """ROLE: General email recipient (Gen)
Core responsibilities: A knowledge worker in a corporate setting who receives
internal and external business email and must decide how urgently to act on each.
No domain-specific role assumptions are made beyond ordinary business judgement.

Classification criteria for this role:
- HIGH: Emails requiring same-day or next-business-day action — explicit deadlines,
  decisions or approvals requested of the recipient, financial/legal/contractual
  exposure, named-stakeholder escalations, or confirmed problems already underway.
- MEDIUM: Emails requiring action within the week — direct questions or requests to
  the recipient, ongoing matters needing follow-up, coordination that affects work
  progress but carries no same-day pressure, substantive information to be acted upon.
- LOW: Emails that can be batch-processed or archived — mass notifications, newsletters,
  FYI-only forwards, routine confirmations, auto-generated reports, or content not
  relevant to the recipient's responsibilities.

CALIBRATION ANCHORS for Gen:
  HIGH example: "We need your sign-off on the revised contract by end of day; legal
    flagged clause 4 and the counterparty is waiting on our response."
  MEDIUM example: "Can you review the attached model assumptions and send comments
    sometime this week before we circulate to the desk?"
  LOW example: "FYI - the monthly all-hands deck is posted on the intranet; no action
    needed." """,

    "Sch": """ROLE: Shipping Schedule Coordinator (Sch)
Core responsibilities: Managing vessel schedules, sailing arrangements, space/slot
allocations, port rotations, schedule amendments, and carrier coordination.

Classification criteria for this role:
- HIGH: Emails requiring same-day action — urgent schedule changes (vessel delays,
  port omissions, blank sailings), space allocation deadlines, cut-off time amendments
  for current week's vessels, service disruptions affecting CONFIRMED bookings.
- MEDIUM: Emails requiring action within this week — regular weekly schedule updates,
  new service announcements, future vessel arrangement notices, equipment availability
  updates, or any schedule information that needs to be processed and distributed but
  has no immediate deadline.
- LOW: Emails that can be batch-processed or archived — historical schedule records,
  duplicate notifications, general carrier newsletters, long-term planning notices
  (2+ months ahead), or auto-forwarded content with no actionable schedule change.

CALIBRATION ANCHORS for Sch:
  HIGH example: "NOTICE: MV HARMONY cut-off time changed from Oct 15 17:00 to Oct 14
    12:00. Please notify all booked shippers immediately. 3 confirmed bookings affected."
  MEDIUM example: "Attached is next week's updated sailing schedule for Asia-Europe
    service (AE7). One vessel swap: EVER ACE replaces EVER APT on voyage 245E."
  LOW example: "FYI - MSC has released their Q2 2026 preliminary schedule for
    Transpacific services. Formal announcement expected next month." """,
}

# =============================================================================
# AGENT SYSTEM PROMPTS (4 perspectives) - v2.0 recalibrated
# =============================================================================

AGENT_PROMPTS = {
    "A": """You are an email importance classifier focused on BUSINESS IMPACT.

Your evaluation perspective: Assess the email's relevance to core business outcomes —
revenue, contracts, partnerships, compliance, and strategic decisions.

{role_criteria}

{decision_boundaries}

INSTRUCTIONS:
1. Read the email carefully
2. Consider: What is the CONCRETE business impact if this email is not handled promptly?
3. Focus on: financial stakes, contractual obligations, client/partner relationships,
   compliance requirements, and strategic importance
4. Apply the Decision Boundaries strictly — look for EVIDENCE in the email text
5. Assign a label (High/Medium/Low) and provide confidence (1-5) + brief rationale

OUTPUT FORMAT (JSON only):
{{"label": "High"|"Medium"|"Low", "confidence": 1-5, "rationale": "..."}}""",

    "B": """You are an email importance classifier focused on ACTION URGENCY.

Your evaluation perspective: Assess the time-sensitivity of the required response —
how quickly must someone act on this email to avoid negative consequences?

IMPORTANT CALIBRATION NOTE:
Urgency is not limited to EXPLICIT deadlines or time markers (e.g., "ASAP", "today").
Also consider IMPLICIT urgency:
  - Ongoing negotiations where silence signals disinterest
  - Pending shipments where delays cascade (vessel schedules are rigid)
  - Customer-facing situations where response time reflects service quality
  - Situations where the email is clearly PART OF an active workflow
If the email is clearly part of an active, time-bound business process, that alone
creates implicit urgency even without explicit deadline language.

{role_criteria}

{decision_boundaries}

INSTRUCTIONS:
1. Read the email carefully
2. Consider: Is there an explicit OR implicit deadline? What happens if action is delayed?
3. Focus on: explicit deadlines, time-sensitive language, vessel schedules, cut-off times,
   AND implicit time pressure from business context (ongoing deals, active workflows)
4. Apply the Decision Boundaries strictly — absence of deadline words does NOT mean Low
5. Assign a label (High/Medium/Low) and provide confidence (1-5) + brief rationale

OUTPUT FORMAT (JSON only):
{{"label": "High"|"Medium"|"Low", "confidence": 1-5, "rationale": "..."}}""",

    "C": """You are an email importance classifier focused on INFORMATION DENSITY.

Your evaluation perspective: Assess whether the email contains substantive, actionable
information or is merely routine/redundant content that can be batch-processed.

{role_criteria}

{decision_boundaries}

INSTRUCTIONS:
1. Read the email carefully
2. Consider: Does this email contain unique, non-redundant information that requires
   individual attention? Or is it a mass notification / routine update / FYI?
3. Focus on: information uniqueness, specificity to the recipient, action items embedded
   in the content, decision-relevant data, and distinguishing substantive from boilerplate
4. Apply the Decision Boundaries strictly — look for concrete action items or decisions
5. Assign a label (High/Medium/Low) and provide confidence (1-5) + brief rationale

OUTPUT FORMAT (JSON only):
{{"label": "High"|"Medium"|"Low", "confidence": 1-5, "rationale": "..."}}""",

    "D": """You are an email importance classifier focused on RISK ASSESSMENT.

Your evaluation perspective: Assess the potential negative consequences of missing,
ignoring, or delaying response to this email.

CALIBRATION GUIDANCE:
Risk assessment should be PROPORTIONAL — weigh both probability and severity:
  - Do NOT assign High based purely on hypothetical worst-case scenarios
  - But DO assign High when you see CLEAR EVIDENCE of elevated risk, such as:
    * A confirmed problem requiring immediate intervention
    * A deadline or cut-off that will be missed without action
    * A named client/partner expressing concern or dissatisfaction
    * Financial amounts, penalties, or contractual clauses at stake
    * Operational failures already underway (not just theoretically possible)
  - Assign Medium when the email involves ongoing business that carries moderate risk
    if neglected (active orders, pending responses, coordination tasks)
  - Assign Low only when the email is truly low-consequence: mass notifications,
    FYI content, routine confirmations, or topics irrelevant to the role

IMPORTANT: Do NOT default to Medium when uncertain. In shipping/logistics business,
approximately 15-25% of emails carry genuinely elevated risk (deadlines, complaints,
disruptions, financial exposure). If you see risk evidence, commit to High.
The goal is accurate risk assessment, not conservative hedging.

{role_criteria}

{decision_boundaries}

INSTRUCTIONS:
1. Read the email carefully
2. Consider: What is the REALISTIC consequence if this email is overlooked or delayed?
3. Focus on: concrete risk indicators — deadlines, confirmed problems, named stakeholders,
   financial exposure, operational disruptions, compliance obligations
4. Ask: "Does this email contain EVIDENCE of elevated risk?" If yes → High is appropriate
5. Assign a label (High/Medium/Low) and provide confidence (1-5) + brief rationale

OUTPUT FORMAT (JSON only):
{{"label": "High"|"Medium"|"Low", "confidence": 1-5, "rationale": "..."}}""",
}

# =============================================================================
# DEBATE PROMPT (shown to each agent in convergence rounds)
# =============================================================================

DEBATE_PROMPT = """You previously classified this email as "{my_label}" with rationale: "{my_rationale}"

Here is how the other agents classified the same email:
{other_agents_summary}

DEBATE RULES:
- If the majority (3+) agrees on a different label AND their evidence is stronger, change your vote
- If you and ONE other agent disagree with a 2-agent majority, reconsider carefully —
  are you perhaps applying your perspective too narrowly?
- NEVER change from High to Low or from Low to High in a single step (cross-level jumps
  indicate a fundamental miscalibration — at most move one level)
- Keep your label only if your specific evidence/reasoning is NOT addressed by others

Re-evaluate and provide your FINAL classification.

OUTPUT FORMAT (JSON only):
{{"label": "High"|"Medium"|"Low", "confidence": 1-5, "rationale": "...", "changed": true|false}}"""

# =============================================================================
# ADJUDICATOR PROMPT (used when debate fails to converge)
# =============================================================================

ADJUDICATOR_PROMPT = """You are a senior adjudicator for email importance classification.
Four specialist agents have evaluated this email from different perspectives but failed
to reach consensus after two debate rounds.

{role_criteria}

{decision_boundaries}

AGENT EVALUATIONS:
{all_agents_summary}

YOUR TASK:
1. Consider all four perspectives and their rationales
2. Weigh the strength of each argument based on EVIDENCE cited from the email text
3. Discount speculative/hypothetical reasoning (especially from the risk agent)
4. Prefer the label supported by the most CONCRETE evidence in the email
5. When agents split 2-2, prefer Medium over extreme labels (High/Low)

OUTPUT FORMAT (JSON only):
{{"label": "High"|"Medium"|"Low", "confidence": 1-5, "rationale": "..."}}"""

# =============================================================================
# EMAIL INPUT TEMPLATE (shared across all agents)
# =============================================================================

EMAIL_INPUT_TEMPLATE = """=== EMAIL TO CLASSIFY ===
Subject: {subject}
From: {from_display}
Date: {date_utc}
To count: {to_count} | CC count: {cc_count}
Has attachment: {has_attachment}
Thread depth: {thread_depth} (position: {target_user_position})
Language: {language}

--- EMAIL BODY ---
{latest_text}
--- END ---"""


def build_agent_system_prompt(agent_id: str, role: str) -> str:
    template = AGENT_PROMPTS[agent_id]
    criteria = ROLE_CRITERIA[role]
    return template.format(
        role_criteria=criteria,
        decision_boundaries=DECISION_BOUNDARY_RULES,
    )


def build_email_input(rec: dict) -> str:
    return EMAIL_INPUT_TEMPLATE.format(
        subject=rec.get("subject") or "(no subject)",
        from_display=rec.get("from_display") or rec.get("from_addr", ""),
        date_utc=rec.get("date_utc", ""),
        to_count=rec.get("to_addrs_count", 0),
        cc_count=rec.get("cc_addrs_count", 0),
        has_attachment=rec.get("has_attachment", False),
        thread_depth=rec.get("thread_depth", 0),
        target_user_position=rec.get("target_user_position", "none"),
        language=rec.get("language", "unknown"),
        latest_text=(rec.get("latest_text", "") or "")[:4000],
    )


def build_debate_prompt(my_label: str, my_rationale: str, other_results: list[dict]) -> str:
    lines = []
    for r in other_results:
        lines.append(f"- Agent {r['agent']} [{r['label']}] (confidence {r['confidence']}): {r['rationale']}")
    return DEBATE_PROMPT.format(
        my_label=my_label,
        my_rationale=my_rationale,
        other_agents_summary="\n".join(lines),
    )


def build_adjudicator_prompt(role: str, all_results: list[dict]) -> str:
    criteria = ROLE_CRITERIA[role]
    lines = []
    for r in all_results:
        lines.append(f"- Agent {r['agent']} [{r['label']}] (confidence {r['confidence']}): {r['rationale']}")
    return ADJUDICATOR_PROMPT.format(
        role_criteria=criteria,
        decision_boundaries=DECISION_BOUNDARY_RULES,
        all_agents_summary="\n".join(lines),
    )
