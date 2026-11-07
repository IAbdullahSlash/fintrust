# FinTrust governance core

Run `python -m unittest -v` from this directory. Requires Python 3.10+ and only its standard library. Fifteen constructed-input tests pass; see test_results.txt. The default thresholds match the proposed binary reference policy; tests use smaller windows to exercise transitions.

This is not an automated trading bot. It has no network access, credentials, broker, model training, portfolio optimization or real-market dataset. It implements a single-task governance subset, fixed decay and in-memory state. Call advance before processing timestamped inputs. The caller must supply fresh candidate forecasts to policy(); forecast ingestion and policy publication are not yet integrated into a durable event service. Retain checkpoints independently to detect log truncation. Hashes without protected checkpoints are not immutable storage. authorize() only guards the current committed policy, expiry and halt; it is not a complete order-risk check.

The experiment protocol in the paper is proposed; no financial performance is measured by these tests.
