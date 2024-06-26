import pytest

@pytest.fixture
def sdata():
    # Create a mock spatialdata object for testing
    # Replace this with your own implementation or use a library like unittest.mock
    return ...

def test_group_gene_expression(sdata):
    # Define test inputs
    genes = ['gene1', 'gene2', 'gene3']
    group = 'group1'
    bin_size = 8
    quantile = 0.70

    # Call the function under test
    result = group_gene_expression(sdata, genes, group, bin_size, quantile)

    # Perform assertions on the result
    assert isinstance(result, pd.DataFrame)
    assert group in result.columns
    assert len(result) == len(sdata.tables[f"square_00{bin_size}um"].obs)

    # Add more assertions as needed
    ...