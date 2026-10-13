# Problem card

| | |
|---|---|
| **Question** | At the moment a customer places an order, will it arrive on a calendar day after the promised delivery date? |
| **Target** | `is_late` = `order_delivered_customer_date.normalize() > order_estimated_delivery_date.normalize()` (binary) |
| **One row is** | One delivered order (`order_id`) placed Jan 2017 to Aug 2018: 96,204 rows |
| **Predict at** | Purchase time, when the customer sees the promised date at checkout |
| **Allowed information** | `window_days` (promised date minus purchase day), `weekday`, `hour`, `customer_state` (encoded as `state_rate`, computed on train only) |
| **Left out** | Everything after purchase: `order_approved_at`, carrier handoff, actual delivery timestamp and `days_to_deliver` (leak: PR AUC 0.90), order status. Undelivered orders (3,000 or so, no label), and months outside Jan 2017 to Aug 2018 (too sparse). The other seven Olist tables are unused this week |
| **Positive rate** | 6.8% late (train 6.8%, test 6.8%; split 76,963 / 19,241, stratified) |
| **Decision it supports** | Flag at-risk orders at checkout so ops can prioritise handling or escalate with the carrier, or pad the promised date on risky routes |
| **Costly mistake** | A missed late order (false negative): unhappy customer, bad review. False positives cost wasted intervention or a longer promise that hurts conversion, but precision near 0.12 means most flags are false alarms. *Adjust if your framing weights these differently.* |
| **Metric** | PR AUC (primary; random = 0.068), with precision and recall at the chosen threshold. Not accuracy: always-on-time scores 0.932 |
| **Baseline to beat** | Always on time: PR AUC 0.068, recall 0. High-risk-state rule: PR AUC 0.102, precision 0.118, recall 0.394. Depth-2 tree: PR AUC 0.102, precision 0.131, recall 0.470 |

## Changes

- *2026-10-13: initial card. Target uses the calendar-day definition, since every promised date is at 00:00 and the timestamp version labels 1,291 same-day deliveries as late (8.1% vs 6.8%).*
- *2026-10-13: random split used for now. Late rate swings from about 1% to 19% by month, so a time-based split is the likely fix in Week 2.*