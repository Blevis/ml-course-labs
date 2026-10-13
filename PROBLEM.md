# Problem card

| | |
|---|---|
| **Question** | At the moment an order is placed, will it arrive on a calendar day after its promised delivery date? |
| **Target** | `is_late` = `delivered_date.normalize() > estimated_delivery_date.normalize()` (binary, calendar-day definition) |
| **One row is** | One delivered order (`order_id`; one `customer_id` per order) |
| **Predict at** | Purchase time, when the customer is shown the promised date |
| **Allowed information** | Purchase timestamp (`weekday`, `hour`), promised date via `window_days` (promise minus purchase day), `customer_state`, and `state_rate` (late rate per state, computed on train only) |
| **Left out** | Everything after purchase: delivery timestamps, carrier handoff, approval time, order status, review score. Test-set labels (never used to build `state_rate` or `high_risk`) |
| **Positive rate** | TODO: fill from `train["is_late"].mean()` and `test["is_late"].mean()` (low base rate, so accuracy is misleading) |
| **Decision it supports** | Flag at-risk orders at checkout so ops can intervene (prioritise handling, escalate to the carrier, warn the customer) or pad the promised date for high-risk routes |
| **Costly mistake** | False negative: a late order that went unflagged, leading to an unhappy customer and a bad review. False positives cost less (wasted intervention, or a padded promise that hurts conversion), but a very low precision makes the flag useless. *Adjust if your framing weights these differently.* |
| **Metric** | Average precision (primary, since it handles the low base rate) and ROC AUC (secondary), both from the continuous score. Precision and recall at the chosen threshold reported alongside. Not accuracy |
| **Baseline to beat** | (1) Base rate / all-zeros predictor. (2) High-risk-state rule: states with train late rate ≥ 1.5 × average, scored by `state_rate`. The depth-2 tree must beat the rule on AP to justify `window_days`/time features |

## Changes

- *2026-10-13: initial card. Target uses the calendar-day definition, not raw timestamp comparison, because every promised date sits at 00:00 and the timestamp version would label same-day afternoon deliveries as late.*