from django.db import models
import json


class TradingDay(models.Model):
    """
    Stores one real trading day of pre-market data + the actual NQ outcome.
    This is the real training data that replaces the synthetic dataset.
    """
    MODE_CHOICES = [
        ('1h', '1 Hour'),
        ('4h', '4 Hour'),
        ('1d', '1 Day'),
        ('general', 'General'),
    ]
    OUTCOME_CHOICES = [
        ('', 'Not set yet'),
        ('Bullish', 'Bullish ▲ (NQ closed above open)'),
        ('Neutral', 'Neutral ↔ (NQ stayed flat)'),
        ('Bearish', 'Bearish ▼ (NQ closed below open)'),
    ]

    date             = models.DateField()
    mode             = models.CharField(max_length=10, choices=MODE_CHOICES, default='1h')

    # Raw % returns entered by user — stored as JSON for flexibility
    # e.g. {"MSFT_1h_ret": -0.16, "NVDA_1h_ret": -0.65, ...}
    raw_inputs       = models.JSONField(default=dict)

    # What the model predicted
    model_prediction = models.CharField(max_length=30, blank=True)
    model_confidence = models.FloatField(default=0.0)   # highest probability %

    # What NQ actually did — user fills this in AFTER market close
    actual_outcome   = models.CharField(max_length=10, choices=OUTCOME_CHOICES, blank=True, default='')

    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date', '-created_at']
        verbose_name = 'Trading Day'
        verbose_name_plural = 'Trading Days'

    def __str__(self):
        return f"{self.date} | {self.mode.upper()} | Pred:{self.model_prediction} | Actual:{self.actual_outcome or '?'}"

    @property
    def is_complete(self):
        """True if actual outcome has been entered."""
        return bool(self.actual_outcome)

    @property
    def was_correct(self):
        """True if model prediction matched actual outcome."""
        if not self.is_complete:
            return None
        return self.model_prediction == self.actual_outcome
