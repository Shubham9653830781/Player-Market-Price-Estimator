
def test_money_conversion():
    from src.data_preprocessing import money_to_num
    assert money_to_num('€1.2M') == 1200000
    assert money_to_num('€800K') == 800000
