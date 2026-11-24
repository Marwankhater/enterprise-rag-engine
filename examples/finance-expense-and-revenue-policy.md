# Acme Inc. Finance Policy: Revenue, Refunds, Discounts and Month-End Close

_Sample content for demonstrating retrieval-augmented generation. Acme Inc. and every figure below are fictional._
_Owner: Finance. Classification: confidential, finance department only. Last updated: 2026._

## 1. Purpose and Scope

This policy explains how Finance records revenue from Acme Cloud subscriptions, how refunds and
cancellations are accounted for, who may approve a discount, and how the monthly close runs. It
applies to all Acme Cloud plans (Starter, Pro and Enterprise), add-ons and services. Customer-facing
terms are published in the product FAQ and the company handbook. Where this policy and a signed
Enterprise agreement differ, the signed agreement governs the accounting for that customer.

Questions about this policy go to the Controller. Exceptions require written approval from the
VP Finance and are logged in the exceptions register.

## 2. Revenue Recognition

Acme follows the principles of ASC 606: revenue is recognized when the service is delivered, not
when cash is received. Each order line is a separate performance obligation.

### 2.1 Monthly plans

Monthly Pro and Enterprise subscriptions are billed at the start of each billing month. The full
monthly charge is recognized as revenue over that month of service. Upgrades are prorated for the
current billing period, so the prorated amount is recognized over the remaining days of that
period. Downgrades take effect at the start of the next billing cycle and need no adjustment.

### 2.2 Annual plans

Annual billing is charged upfront at 10 times the monthly price, which is how customers save two
months. The full invoice amount is booked to account 2400 Deferred Revenue on the invoice date and
released to account 4000 Subscription Revenue in 12 equal monthly amounts. Example: a Pro customer
with 10 seats on annual billing pays 10 x $290 = $2,900, and Finance recognizes $241.67 per month
for 12 months.

### 2.3 Add-ons and services

Recurring add-ons (extra storage, premium support, SSO for Pro, extended audit log retention)
follow the billing frequency of the plan they are attached to and are recognized the same way.
The guided onboarding package is a one-time service and is recognized in full when the customer
signs the onboarding completion form, not on the invoice date.

### 2.4 Starter plan

The Starter plan is free. Starter orders appear in the sales database with a unit price of $0.00
and generate no revenue. Paid add-ons on a Starter account are recognized as in section 2.3.

## 3. Refund and Cancellation Accounting

The customer-facing rules are set by the handbook: a full refund is available within 30 days of
purchase, refunds go back to the original payment method within 5 to 7 business days, and after
30 days refunds are offered as store credit only. Annual plans cancelled within the first 14 days
receive a full refund. After 14 days, the customer receives a prorated refund of unused months.

Finance records these cases as follows:

- **Refunded orders**: payment was collected and then returned. The refund is posted to account
  4900 Refunds and Credits (a contra-revenue account) in the period the refund is approved. Any
  unreleased balance in Deferred Revenue for that order is reversed at the same time.
- **Store credit**: credit issued after the 30-day window is a liability, posted to account 2450
  Customer Credits, and released when the customer uses it.
- **Cancelled orders**: the order was voided before payment was collected. No revenue and no
  refund are recorded. Cancelled orders stay in the sales database for audit purposes.

Net revenue reports count only orders with status paid. Refunded and cancelled orders are reported
separately so the refund rate can be tracked each month.

## 4. Discount Approval Matrix

Discounts are measured as a percentage off the list price of the order. List prices are: Starter
$0, Pro $29 per user per month, and Enterprise $79 per user per month as the list reference for
quotes. The Pro list price is not negotiated; Pro discounts are limited to the standard programs
below.

| Discount off list price | Approver | Required documentation |
|---|---|---|
| Up to 10% | Account Executive | Note in the CRM opportunity |
| 11% to 20% | Head of Sales | Deal desk form |
| 21% to 30% | VP Finance | Deal desk form and written business case |
| Above 30% | CFO and CEO jointly | Business case and margin analysis |

Standard programs do not need approval because they are already priced into the plan catalog:

- Annual billing (two months free compared to monthly billing).
- 50% off for registered nonprofits, after verification.
- Free Pro access for accredited students.

Discounts stack additively, not multiplicatively, when the approval level is determined. A 15%
negotiated discount on an annual contract therefore needs Head of Sales approval: the annual billing
saving is a standard program and does not count toward the threshold. Approvals must be recorded
before the order form is sent to the customer. An order form sent without the required approval is
not booked until the approval is obtained.

## 5. Expense Controls

The handbook sets the employee-facing rules: receipts are required for any expense over $25,
expenses are reimbursed within two weeks of submission, and submissions more than 60 days after the
expense was incurred need manager approval. Finance applies these additional controls:

| Expense amount | Approval required |
|---|---|
| Up to $2,500 | Direct manager |
| $2,501 to $10,000 | Department head |
| Above $10,000 | VP Finance, with a purchase order |

The annual home-office budget ($1,000) and learning budget ($1,500) are tracked per employee in the
expense system. Requests above either budget are rejected automatically. Offsite meals are
reimbursed up to $75 per day.

## 6. Month-End Close Calendar

The books close on business day 6 (BD6) of the following month. Business days exclude weekends
and the 10 company holidays.

| Business day | Task | Owner |
|---|---|---|
| BD1 | Billing cut-off, subscription revenue run, deferred revenue release | Revenue Accountant |
| BD2 | Bank and payment processor reconciliations (cards, ACH and SEPA) | Accounting Team |
| BD3 | Refund and store credit reconciliation, deferred revenue roll-forward | Revenue Accountant |
| BD4 | Accruals, expense report cut-off, payroll journal | Accounting Team |
| BD5 | Flux review of revenue and expenses against the prior month | Controller |
| BD6 | Close sign-off and lock of the period | Controller and VP Finance |
| BD8 | Management reporting pack distributed | FP&A |

Expense reports submitted after BD4 are posted to the next period. Any journal entry after BD6
requires Controller approval and a note in the close checklist.

## 7. Record Keeping

Invoices, refund approvals, discount approvals and close checklists are kept for seven years, in
line with the company data retention policy. Finance reviews access to the accounting system each
quarter, following the principle of least privilege described in the handbook.
