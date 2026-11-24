# Acme Engineering Security Incident Runbook

_Sample content for demonstrating retrieval-augmented generation. Acme Inc., its tools and processes are fictional._
_Owner: Security team. Classification: internal, engineering department. Last updated: 2026._

## 1. When to Use This Runbook

Use this runbook for any suspected security incident: a lost or stolen device, a phishing link that
was clicked, an exposed key or credential, unexpected access to customer data, or malware on a
company laptop. It also applies to production outages that may have a security cause.

The handbook rule applies to everyone: report a suspected security incident to the security team
within one hour of discovery. Reporting in good faith is never penalized, and a false alarm is
always better than a late report.

## 2. Severity Levels

The first responder assigns a severity as soon as the incident is declared. The severity can be
raised or lowered as facts become clear.

| Severity | Definition | Examples | Acknowledge | Status updates |
|---|---|---|---|---|
| SEV1 | Critical: confirmed breach of customer data, production down for all customers, or production credentials exposed publicly | Leaked database credentials in a public repository | Page acknowledged within 15 minutes, 24/7 | Every 30 minutes |
| SEV2 | Major: suspected unauthorized access, or a core feature down for many customers or for one Enterprise customer | Unusual admin logins from an unknown country | Page acknowledged within 15 minutes, 24/7 | Every 60 minutes |
| SEV3 | Minor: limited impact and a workaround exists | Lost laptop with full-disk encryption enabled | Within 4 business hours | Once per business day |
| SEV4 | Low: no customer impact | Phishing email reported without a click | Next business day | When resolved |

SEV1 and SEV2 incidents page the on-call engineer. SEV3 and SEV4 incidents are tracked as tickets in
the security queue and do not page anyone outside business hours.

## 3. Reporting an Incident

1. Post in the #incidents Slack channel with what you saw, when, and which systems are involved.
   If Slack is unavailable, use the security intake form or call the on-call phone number.
2. Do not try to investigate on your own, delete evidence, or reboot an affected machine.
3. If you clicked a phishing link or entered your password, change the password from a different
   device and tell the responder which accounts are affected.
4. If a device is lost or stolen, report it so the security team can trigger a remote wipe.

## 4. On-Call Paging

Engineers join the on-call rotation after their first month. Each shift runs one week, from Monday
10:00 to the following Monday 10:00 in the primary's local time, and on-call shifts are
compensated. Each shift has a primary and a secondary responder.

- The primary must acknowledge a page within 15 minutes.
- If the primary does not acknowledge within 15 minutes, the page escalates to the secondary.
- If neither acknowledges within 30 minutes of the first page, the page escalates to the
  engineering manager on duty.
- Acknowledging a page means taking ownership of the first response, not fixing everything alone.
  Escalate early and often.

Handing over a shift requires a short written note in the #oncall channel listing open incidents
and anything that may page during the next week.

## 5. Escalation Steps

1. **Acknowledge and assess.** The on-call engineer acknowledges the page, confirms the report and
   assigns a severity.
2. **Declare the incident.** For SEV1 and SEV2, open a dedicated incident channel and name an
   Incident Commander (IC). The IC coordinates; other responders do the hands-on work.
3. **Notify leadership.** For SEV1, the IC notifies the VP Engineering and the Head of Security
   within 30 minutes of declaration. For SEV2, notify the Head of Security within 60 minutes.
4. **Contain.** Rotate exposed credentials through the secrets manager, revoke sessions, block
   malicious addresses and isolate affected hosts. Rollbacks are preferred over a forward-fix under
   pressure.
5. **Involve Legal and Support.** If customer data may be affected, the IC brings in Legal to decide
   on notification duties. Support informs affected Enterprise customers; critical production
   issues for Enterprise customers receive a first response within one hour, 24/7, under the
   support SLA.
6. **Recover.** Restore service, confirm that the attacker no longer has access and monitor closely
   for 72 hours.
7. **Close.** The IC closes the incident when impact has ended, and schedules the postmortem.

## 6. Evidence Handling

Preserve logs, screenshots and timelines before making changes where possible. Store evidence in
the restricted incident folder, not in public channels. Security incident records and postmortems
are kept for three years under the data retention policy.

## 7. Postmortems

Every incident is followed by a blameless postmortem focused on systems, not individuals. For SEV1
and SEV2, the postmortem document is due within five business days of closing the incident and is
reviewed at the weekly engineering review. For SEV3, a short written summary is enough. SEV4
incidents need no postmortem.

### 7.1 Postmortem template

| Section | What to write |
|---|---|
| Summary | Two or three sentences: what happened and the final severity |
| Impact | Customers, data and systems affected, and for how long |
| Timeline | Key events in UTC, from first signal to resolution |
| Detection | How the incident was found and how long detection took |
| Root cause | The technical and process causes, without naming individuals as the cause |
| What went well | Practices that limited impact |
| What went poorly | Gaps in tooling, alerts or documentation |
| Action items | Each item with an owner, a ticket and a due date |

Action items from SEV1 postmortems are reviewed every week until they are closed.

## 8. Quick Reference

- Report suspected incidents within one hour of discovery.
- Acknowledge SEV1 and SEV2 pages within 15 minutes.
- SEV1 status updates every 30 minutes.
- Blameless postmortem within five business days for SEV1 and SEV2.
