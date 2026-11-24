# Acme Cloud Sales Playbook

_Sample content for demonstrating retrieval-augmented generation. Acme Inc., Acme Cloud and all prices are fictional._
_Owner: Sales. Classification: internal, sales department. Last updated: 2026._

## 1. How to Use This Playbook

This playbook is the shared reference for Account Executives, Sales Development Representatives and
Solutions Engineers. It covers how to position each plan, how Enterprise deals move through the
pipeline, what discounts you can offer on your own, and how to answer common objections. Prices and
plan features must always match the public product FAQ. If you find a difference, the FAQ is
correct and this playbook should be updated.

## 2. Plans and Positioning

| Plan | Price | What is included | Best fit |
|---|---|---|---|
| Starter | Free | 1 project, 1 GB of storage, community support | Individuals and evaluations |
| Pro | $29 per user per month | Unlimited projects, 100 GB of storage, priority email support, full API | Teams of 3 to 60 users |
| Enterprise | Custom-priced (list reference $79 per user per month) | SSO, dedicated support engineer, 99.9% uptime SLA, on-premise deployment options, customer-managed encryption keys | Regulated industries and teams above 50 users |

Annual billing costs 10 times the monthly price, so the customer saves two months. Lead with annual
billing for every Pro and Enterprise opportunity.

### 2.1 Positioning by plan

- **Starter** is a way in, not a revenue line. Help Starter users reach the point where they need a
  second project or more storage, then offer Pro.
- **Pro** sells on productivity: unlimited projects, the full API with 600 requests per minute, and
  webhooks. Pro list price is fixed; do not negotiate it.
- **Enterprise** sells on control and risk reduction: SSO, a dedicated support engineer, the uptime
  SLA, and data residency in the US or EU region.

### 2.2 Add-ons

| Add-on | Price | Available on |
|---|---|---|
| Extra storage (100 GB block) | $10 per block per month | All plans |
| Premium support (24/7 phone) | $199 per month | Pro |
| Single sign-on (SSO) for Pro | $6 per user per month | Pro (SSO is included in Enterprise) |
| Extended audit log retention | $49 per month | Enterprise |
| Guided onboarding package | $1,500 one-time | Pro and Enterprise |

Recurring add-ons follow the plan's billing frequency, so on annual billing they also cost 10 times
the monthly price.

### 2.3 Payment terms

All plans accept major credit cards. Annual Enterprise plans can also pay by bank transfer (ACH and
SEPA), and invoicing is available for Enterprise customers. Monthly Enterprise plans pay by card.

## 3. Enterprise Deal Stages

Every Enterprise opportunity in the CRM must be in exactly one stage. Move a deal forward only when
the exit criteria are met.

| Stage | Name | Exit criteria | Typical length |
|---|---|---|---|
| 0 | Qualified lead | Budget owner identified, more than 50 potential users | 1 week |
| 1 | Discovery | Pain points, decision process and timeline documented | 2 weeks |
| 2 | Technical evaluation | Pilot of up to 30 days completed with agreed success criteria | 2 to 5 weeks |
| 3 | Proposal | Order form with pricing sent, discount approved where needed | 1 to 2 weeks |
| 4 | Security and legal review | Security questionnaire answered, contract redlines agreed | 2 to 4 weeks |
| 5 | Closed won or closed lost | Signed order form, or loss reason recorded | None |

Solutions Engineers own stage 2. For stage 4, send the standard security package first (SOC 2 Type
II report, encryption overview, data retention policy) before answering custom questionnaires.

## 4. Discount Rules

Discounts apply to Enterprise deals. They are measured as a percentage off the list reference price
and follow the discount approval matrix owned by Finance:

| Discount | Who approves |
|---|---|
| Up to 10% | You (the Account Executive), with a note in the CRM |
| 11% to 20% | Head of Sales |
| 21% to 30% | VP Finance |
| Above 30% | CFO and CEO jointly |

Rules that apply to every deal:

- Get the approval before the order form goes to the customer. Finance does not book orders sent
  without the required approval.
- Standard programs do not need approval: annual billing (two months free), 50% off for registered
  nonprofits, and free Pro access for accredited students.
- Do not offer discounts on Pro. If price is the blocker on Pro, offer annual billing.
- Refund terms are not a negotiation tool. The standard refund policy applies unless the signed
  Enterprise agreement says otherwise.

## 5. Objection Handling

**"It is too expensive."** Start with annual billing, which saves two months. Then quantify the time
saved by unlimited projects and the full API. For Enterprise, compare the cost of the dedicated
support engineer and the 99.9% uptime SLA with the customer's cost of downtime.

**"Where is our data stored?"** Data is stored in SOC 2 Type II certified data centers. The customer
chooses the US or EU region when creating a project. Be clear that the region cannot be changed after
creation, so the choice should be made with their compliance team.

**"We are worried about lock-in."** A full export in JSON or CSV is available from the dashboard at
any time, on every plan including the free tier. There is no lock-in.

**"Is it secure?"** Data is encrypted in transit with TLS 1.2+ and at rest with AES-256.
Enterprise customers can use customer-managed encryption keys.

**"Your API limits are too low."** Pro allows 600 requests per minute. Enterprise limits are
negotiated per contract. Clients that exceed the limit receive HTTP 429 and should retry with
exponential backoff.

**"What if something breaks?"** Enterprise customers get 24/7 support, a private Slack channel and a
dedicated support engineer. Critical production issues receive a first response within one hour.
Pro customers get priority email support with a four-business-hour first response.

**"Can we cancel?"** Yes. Customers can cancel at any time from Settings > Billing. Cancellation
stops future charges. Annual plans cancelled within the first 14 days receive a full refund.

## 6. Handover to Customer Success

When a deal is closed won, the Account Executive completes the handover form within two business
days: contacts, success criteria from the pilot, agreed discount and any non-standard terms.
Enterprise customers are introduced to their dedicated support engineer in the kickoff meeting.
