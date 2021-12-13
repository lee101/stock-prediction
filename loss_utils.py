import numpy as np

# TRADING_FEE = 0.0007 # fee actually changes for small trades - this is for 100k
TRADING_FEE = 0.003  # fee actually changes for small trades

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_trading_profit(scaler, x_test, y_test, y_test_pred):
    """
    Calculate trading profits
    :param x_test:
    :param y_test:
    :param y_test_pred:
    :return:
    """
    last_values = x_test[:, -1, :].detach().cpu().numpy()
    percent_movements = ((y_test - last_values) / last_values) + 1
    # those where not scaled properly, scale properly for logging purposes
    last_values_scaled = scaler.inverse_transform(last_values)
    percent_movements_scaled = (
        (scaler.inverse_transform(y_test) - last_values_scaled) / last_values_scaled
    ) + 1
    detached_y_test_pred = y_test_pred
    bought_profits = np.clip(detached_y_test_pred, 0, 1) * percent_movements_scaled
    saved_money = 1 - np.abs(detached_y_test_pred)
    current_profit = np.sum(
        # saved money
        saved_money
        +
        # bought
        bought_profits
        +
        # sold
        np.abs((np.clip(detached_y_test_pred, -1, 0) * percent_movements_scaled))
        # fee
        - (np.abs(detached_y_test_pred) * TRADING_FEE)
    ) - len(detached_y_test_pred)
    # todo random deprecation?
    return current_profit


def torch_inverse_transform(scaler, values):
    if not scaler:
        return values
    # copy memory of torch tensor
    values = values.clone()
    values -= torch.tensor(scaler.min_).to(DEVICE)
    values /= torch.tensor(scaler.scale_).to(DEVICE)
    return values


# def calculate_trading_profit_torch(scaler, last_values, y_test, y_test_pred):
#     """
#     Calculate trading profits
#     :param last_values:
#     :param y_test:
#     :param y_test_pred:
#     :return:
#     """
#     percent_movements = ((y_test - last_values) / last_values) + 1
#     # those where not scaled properly, scale properly for logging purposes
#     last_values_scaled = torch_inverse_transform(scaler, last_values)
#     percent_movements_scaled = (
#         (torch_inverse_transform(scaler, y_test) - last_values_scaled) / last_values_scaled
#     ) + 1
#     bought_profits =((torch.clamp(y_test_pred, 0, 10) * percent_movements_scaled)
#              - (torch.clamp(y_test_pred, 1, 100) - 1)
#              - torch.clamp(y_test_pred, 0, 10))
#     # bought_profits = (
#     #                      (torch.clamp(y_test_pred, 0, 10) * percent_movements_scaled)
#     #                      - (torch.clamp(y_test_pred, 1, 100) - 1)
#     #                  ) - 1
#     saved_money = torch.clamp(
#         1 - torch.abs(y_test_pred), 0, 500
#     )  # only can save positive amount if we dont leverage
#     sold_profits = torch.abs((-torch.clamp(y_test_pred, -10, 0) * (1 - (percent_movements_scaled - 1))))
#     sold_saved_money = torch.clamp(torch.abs(torch.clamp(y_test_pred, -1, 0)), 0, 1)
#     current_profit = torch.sum(
#         # saved money
#         saved_money
#         +
#         # bought
#         bought_profits
#         +
#         # sold
#         sold_profits
#         + sold_saved_money
#         # fee
#         - (torch.abs(y_test_pred) * TRADING_FEE)
#     ) / len(y_test_pred)
#     # todo random deprecation?
#     return current_profi


def calculate_trading_profit_torch(scaler, last_values, y_test, y_test_pred):
    """
    Calculate trading profits
    :param last_values:
    :param y_test:
    :param y_test_pred:
    :return:
    """
    # percent_movements = ((y_test - last_values) / last_values) + 1

    last_values_scaled = torch_inverse_transform(scaler, last_values)
    percent_movements_scaled = (torch_inverse_transform(scaler, y_test) - last_values_scaled) / (
        (torch_inverse_transform(scaler, y_test) + last_values_scaled) / 2
    )  # not scientific
    detached_y_test_pred = y_test_pred
    bought_profits = torch.clip(detached_y_test_pred, 0, 10) * percent_movements_scaled
    sold_profits = torch.clip(y_test_pred, -10, 0) * percent_movements_scaled
    # saved_money = torch.clamp(
    #             1 - torch.abs(y_test_pred), 0, 500
    #         )
    current_profit = torch.sum(
        # saved money
        # saved_money
        # +
        # bought
        bought_profits
        +
        # sold
        sold_profits
        # fee
        - (torch.abs(detached_y_test_pred) * TRADING_FEE)
    ) / len(detached_y_test_pred)
    # todo random deprecation?
    return current_profit


def calculate_trading_profit_no_scale(last_values, y_test, y_test_pred):
    """
    Calculate trading profits
    :param x_test:
    :param y_test:
    :param y_test_pred: how much portfolio was invested - 1s for selling all, 1s for buying all
    :return:
    """
    return calculate_trading_profit_torch(None, last_values, y_test, y_test_pred)
    # percent_movements = ((y_test - last_values) / last_values) + 1
    # # those where not scaled properly, scale properly for logging purposes
    # last_values_scaled = last_values
    #
    # percent_movements_scaled = ((y_test - last_values_scaled) / last_values_scaled) + 1
    # bought_profits = (
    #     (torch.clamp(y_test_pred, 0, 10) * percent_movements_scaled)
    #     - (torch.clamp(y_test_pred, 1, 100) - 1)
    # ) - 1
    # saved_money = torch.clamp(
    #     1 - torch.abs(y_test_pred), 0, 500
    # )  # only can save positive amount if we dont leverage
    # sold_profits = torch.abs((torch.clamp(y_test_pred, -10, 0) * percent_movements_scaled))
    # sold_saved_money = torch.clamp(torch.abs(torch.clamp(y_test_pred, -1, 0)), 0, 1)
    # current_profit = torch.sum(
    #     # saved money
    #     saved_money
    #     +
    #     # bought
    #     bought_profits
    #     +
    #     # sold
    #     sold_profits
    #     + sold_saved_money
    #     # fee
    #     - (torch.abs(y_test_pred) * TRADING_FEE)
    # )
    # # todo random deprecation?
    # return current_profit



def get_trading_profits_list(scaler, last_values, y_test, y_test_pred):
    """
    Calculate trading profits
    :param last_values:
    :param y_test:
    :param y_test_pred:
    :return:
    """
    # percent_movements = ((y_test - last_values) / last_values) + 1

    last_values_scaled = torch_inverse_transform(scaler, last_values)
    percent_movements_scaled = (torch_inverse_transform(scaler, y_test) - last_values_scaled) / (
        (torch_inverse_transform(scaler, y_test) + last_values_scaled) / 2
    )  # not scientific
    detached_y_test_pred = y_test_pred
    bought_profits = torch.clip(detached_y_test_pred, 0, 10) * percent_movements_scaled
    sold_profits = torch.clip(y_test_pred, -10, 0) * percent_movements_scaled
    # saved_money = torch.clamp(
    #             1 - torch.abs(y_test_pred), 0, 500
    #         )
    current_profits = (
        # saved money
        # saved_money
        # +
        # bought
        bought_profits
        +
        # sold
        sold_profits
        # fee
        - (torch.abs(detached_y_test_pred) * TRADING_FEE)
    )

    # todo random deprecation?
    return current_profits
