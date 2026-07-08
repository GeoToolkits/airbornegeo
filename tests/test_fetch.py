from unittest.mock import patch

import pandas as pd
import pytest

from airbornegeo import fetch

# fetch_agap_gravity() downloads a real remote dataset via pooch.retrieve - we
# mock that call rather than hit the network in tests, per pytest-guide's
# recommendation that tests stay fast and hermetic.


def test_fetch_agap_gravity_reads_retrieved_file(tmp_path):
    """The CSV file returned by pooch.retrieve should be read into a DataFrame and returned."""
    csv_path = tmp_path / "AGAP_BAS_Grav.XYZ"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    with patch.object(
        fetch.pooch, "retrieve", return_value=str(csv_path)
    ) as mock_retrieve:
        result = fetch.fetch_agap_gravity()

    assert mock_retrieve.call_count == 1
    assert isinstance(result, pd.DataFrame)
    assert result.columns.tolist() == ["a", "b"]
    assert result.a.tolist() == [1, 3]


def test_fetch_agap_gravity_calls_pooch_with_expected_hash_and_url():
    """pooch.retrieve should be called with the expected dataset URL, filename, and hash."""
    with patch.object(fetch.pooch, "retrieve", return_value=None) as mock_retrieve:
        with pytest.raises(ValueError):  # pd.read_csv(None) raises
            fetch.fetch_agap_gravity()

    _, kwargs = mock_retrieve.call_args
    assert kwargs["url"].startswith("https://ramadda.data.bas.ac.uk/")
    assert kwargs["fname"] == "AGAP_BAS_Grav.XYZ"
    assert kwargs["known_hash"] == (
        "391225810f1d15be21b37f506c098960d92af9b3ec0b48bb55dfa20e7b4cf25e"
    )
