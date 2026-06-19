# -*- coding: utf-8 -*-
import pytest

SCENARIOS = [
    pytest.param((50,  [100.0, 80.0],              0.90), id="2srv-light"),
    pytest.param((150, [100.0, 80.0, 60.0],        0.90), id="3srv-base"),
    pytest.param((200, [200.0, 150.0, 100.0],      0.85), id="3srv-heavy"),
    pytest.param((30,  [50.0, 50.0, 50.0, 50.0],   0.90), id="4srv-equal"),
    pytest.param((80,  [100.0, 100.0],              0.95), id="2srv-highrho"),
]

@pytest.fixture(params=SCENARIOS)
def scenario(request):
    lam, mu_list, rho_max = request.param
    return {"lambda_total": lam, "mu_list": mu_list, "rho_max": rho_max}
