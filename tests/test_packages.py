import pkg_resources

def test_installed_packages():
    # Test that the housing price prediction package is installed or not
    # Get the list of packages
    installed_packages = [
        package.project_name
        for package in pkg_resources.working_set]
    assert 'housingPricePrediction' in installed_packages
