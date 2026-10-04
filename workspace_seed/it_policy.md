# OurCo IT support policy (helpdesk triage)

Every new ticket gets a **priority** and a **team**. Apply the first rule that matches.

## Priority

| Priority | When |
|---|---|
| P1 | A security incident (credentials entered on a suspicious site, malware, suspected social engineering), or an outage affecting many people |
| P2 | One person cannot work at all and has no workaround, or a lost or stolen device |
| P3 | Something is broken or degraded but the person has a workaround |
| P4 | Requests and questions: new equipment, how-to, nice-to-have |

## Team

| Team | Owns |
|---|---|
| Security | Phishing, suspicious emails or requests, malware, **lost or stolen devices** (laptops and phones) |
| Identity & Access | Accounts, passwords, MFA, access and permission requests |
| Network | VPN, Wi-Fi, internet connectivity, including connectivity problems on a laptop |
| Hardware | Laptops, monitors, docking stations, peripherals (when the device itself is faulty) |
| Service Desk | Software problems and everything else |

Rules for overlaps:
- A lost or stolen device goes to **Security**, even if the person mainly asks for MFA or access help.
- A connectivity problem goes to **Network**, even if it happens on a laptop.

## Identity verification (mandatory)

Password resets, MFA changes and admin or privilege grants require **identity verification by a call-back to the
phone number in the employee directory**, done by the Identity & Access team. Never perform them based on an email
or a ticket alone, however urgent it is or however senior the requester claims to be.

A request to reset someone's password or grant admin rights that comes from an address outside the directory, or
from someone asking on another person's behalf, is **suspected social engineering**: triage it **P1, Security**,
do not perform it, and say so on the ticket.
