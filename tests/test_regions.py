"""Tests for the 25 administrative regions (24 oblasts + Kyiv city) and per-oblast isolation."""

import pytest
import pandas as pd

from src.utils.regions import (
    ALL_REGION_SLUGS,
    TOTAL_REGIONS_COUNT,
    filter_by_region,
    get_all_regions,
    get_hromada,
    get_region,
    get_region_hromadas,
    is_valid_hromada,
    is_valid_region,
)


def test_total_regions_count():
    """Verify that exactly 25 administrative regions (24 oblasts + Kyiv city) are registered."""
    assert TOTAL_REGIONS_COUNT == 25
    assert len(ALL_REGION_SLUGS) == 25
    regions = get_all_regions()
    assert len(regions) == 25


def test_all_24_oblasts_and_kyiv_city_present():
    """Verify presence of Kyiv city and key oblasts."""
    expected_slugs = {
        "cherkasy", "chernihiv", "chernivtsi", "dnipropetrovsk", "donetsk",
        "ivano_frankivsk", "kharkiv", "kherson", "khmelnytskyi", "kirovohrad",
        "kyiv_city", "kyiv_oblast", "luhansk", "lviv", "mykolaiv", "odesa",
        "poltava", "rivne", "sumy", "ternopil", "vinnytsia", "volyn",
        "zakarpattia", "zaporizhzhia", "zhytomyr",
    }
    assert set(ALL_REGION_SLUGS) == expected_slugs


def test_unique_region_ids():
    """Verify each region has a unique ID from 1 to 25."""
    regions = get_all_regions()
    ids = [r.id for r in regions.values()]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(range(1, 26))


@pytest.mark.parametrize("slug", ALL_REGION_SLUGS)
def test_individual_oblast_resolution(slug: str):
    """Verify each oblast can be resolved individually by slug and by numeric ID."""
    region = get_region(slug)
    assert region.slug == slug
    assert is_valid_region(slug)
    assert is_valid_region(region.id)
    assert get_region(region.id).slug == slug


def test_invalid_region_lookup():
    """Ensure invalid region raises appropriate error."""
    assert not is_valid_region("non_existent_oblast")
    assert not is_valid_region(999)
    with pytest.raises(KeyError):
        get_region("atlantis")
    with pytest.raises(ValueError):
        get_region(100)


def test_filter_by_region_isolation():
    """Verify that filter_by_region isolates rows strictly for the target oblast."""
    df = pd.DataFrame({
        "region_slug": ["kyiv_city", "kharkiv", "lviv", "kyiv_city"],
        "value": [10, 20, 30, 40],
    })
    filtered = filter_by_region(df, "kyiv_city")
    assert len(filtered) == 2
    assert (filtered["region_slug"] == "kyiv_city").all()
    assert list(filtered["value"]) == [10, 40]

    # Test filtering by numeric ID
    filtered_kharkiv = filter_by_region(df, get_region("kharkiv").id)
    assert len(filtered_kharkiv) == 1
    assert filtered_kharkiv["region_slug"].iloc[0] == "kharkiv"


def test_hromadas_for_all_regions():
    """Verify all 25 regions have registered representative hromadas."""
    for slug in ALL_REGION_SLUGS:
        hromadas = get_region_hromadas(slug)
        assert len(hromadas) >= 1, f"Region {slug} has no hromadas defined!"
        for h in hromadas:
            assert h.slug
            assert h.name_en
            assert h.name_uk


def test_hromada_lookup():
    """Verify specific hromada lookup and validation."""
    assert is_valid_hromada("kharkiv", "kupiansk")
    assert is_valid_hromada("kyiv_oblast", "bucha")
    assert not is_valid_hromada("lviv", "non_existent_hromada")

    h = get_hromada("kharkiv", "kupiansk")
    assert h is not None
    assert h.slug == "kupiansk"
    assert h.name_en == "Kupiansk"
