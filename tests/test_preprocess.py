from src.preprocess import clean_tweet


def test_clean_tweet_replaces_social_media_noise():
    cleaned = clean_tweet("RT @person: Visit https://example.com &amp; learn!")
    assert "RETWEET".lower() in cleaned
    assert "USER".lower() in cleaned
    assert "URL".lower() in cleaned
    assert "&amp;" not in cleaned
