# Error overlap analysis

Test samples: 22080

## Wanxiang vs Tiny-8M

- both correct: 18493
- Wanxiang only: 925
- Tiny-8M only: 633
- both wrong: 2029
- oracle accuracy: 0.908107
- oracle gain over Wanxiang: 0.028668

## Simple hybrid

```json
{
  "test": {
    "contested_samples": 11106,
    "contested_top1": 0.837655321447866,
    "coverage": 0.04375,
    "losses": 166,
    "net_wins": 531,
    "promotion_precision": 0.7215320910973085,
    "samples": 22080,
    "threshold": 0.0495232418179512,
    "top1": 0.9034873188405798,
    "wins": 697
  },
  "tuning": {
    "contested_samples": 11100,
    "contested_top1": 0.8384684684684685,
    "coverage": 0.042962023021843564,
    "losses": 144,
    "net_wins": 528,
    "promotion_precision": 0.7088607594936709,
    "samples": 22066,
    "selected_on": "val",
    "status": "AVAILABLE",
    "threshold": 0.0495232418179512,
    "top1": 0.9030635366627391,
    "wins": 672
  }
}
```
