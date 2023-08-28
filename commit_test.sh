pytest tests/test_operators.py -k task0_1 -q > test_res
pytest tests/test_operators.py -k task0_2 -q >> test_res
pytest tests/test_operators.py -k task0_3 -q >> test_res
pytest tests/test_module.py -k task0_4 -q >> test_res

pytest tests/test_scalar.py -k task1_1 -q >> test_res
pytest tests/test_scalar.py -k task1_2 -q >> test_res
pytest tests/test_autodiff.py -k task1_3 -q >> test_res
pytest tests/test_scalar.py -k task1_4 -q >> test_res
pytest tests/test_autodiff.py -k task1_4 -q >> test_res

pytest tests/test_tensor_data.py -k task2_1 -q >> test_res
pytest tests/test_tensor_data.py -k task2_2 -q >> test_res
pytest tests/test_tensor.py -k task2_3 -q >> test_res
pytest tests/test_tensor.py -k task2_4 -q >> test_res

# pytest tests/test_tensor_general.py -k task3_1 -q >> test_res
