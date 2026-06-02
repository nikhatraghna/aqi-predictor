"""PM2.5 → AQI category mapping boundary tests."""
#from src.inference.forecast_next_3_days import pm25_to_category
from src.inference.forecast_future import pm25_to_category


def test_category_labels():
    assert pm25_to_category(5)[0]   == "Good"
    assert pm25_to_category(20)[0]  == "Moderate"
    assert pm25_to_category(45)[0]  == "Unhealthy (Sensitive)"
    assert pm25_to_category(100)[0] == "Unhealthy"
    assert pm25_to_category(200)[0] == "Very Unhealthy"
    assert pm25_to_category(400)[0] == "Hazardous"


def test_category_boundaries():
    assert pm25_to_category(0)[0]  == "Good"
    assert pm25_to_category(12)[0] == "Moderate"
    assert pm25_to_category(35)[0] == "Unhealthy (Sensitive)"


def test_category_returns_emoji():
    label, emoji = pm25_to_category(50)
    assert isinstance(label, str) and isinstance(emoji, str)
