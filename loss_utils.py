import numpy as np

# TRADING_FEE = 0.0007 # fee actually changes for small trades - this is for 100k
# TRADING_FEE = 0.003  # fee actually changes for small trades
CRYPTO_TRADING_FEE = .0015  # maker fee taker is .0025
# from pytorch_forecasting import MultiHorizonMetric

TRADING_FEE = 0.0005
# equities .0000278

# Try to import torch, but allow graceful fallback for testing
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    DEVICE = None


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
    ) / len(detached_y_test_pred)
    # todo random deprecation?
    return current_profit


def torch_inverse_transform(scaler, values):
    if not scaler:
        return values
    # copy memory of torch tensor
    # values = values.clone()
    # values -= torch.tensor(scaler.min_).to(DEVICE)
    # values /= torch.tensor(scaler.scale_).to(DEVICE)
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

    # last_values_scaled = last_values # no scaling
    # y_test_rescaled = y_test # no scaling
    percent_movements_scaled = y_test  # not scientific
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
    ) / detached_y_test_pred.numel()
    # todo random deprecation?
    return current_profit


def calculate_trading_profit_torch_buy_only(scaler, last_values, y_test, y_test_pred):
    """
    Calculate trading profits
    :param last_values:
    :param y_test:
    :param y_test_pred:
    :return:
    """
    # percent_movements = ((y_test - last_values) / last_values) + 1

    # last_values_scaled = last_values # no scaling
    # y_test_rescaled = y_test # no scaling
    percent_movements_scaled = y_test  # not scientific
    detached_y_test_pred = y_test_pred
    bought_profits = torch.clip(detached_y_test_pred, 0, 10) * percent_movements_scaled
    # sold_profits = torch.clip(y_test_pred, -10, 0) * percent_movements_scaled
    # saved_money = torch.clamp(
    #             1 - torch.abs(y_test_pred), 0, 500
    #         )
    current_profit = torch.sum(
        # saved money
        # saved_money
        # +
        # bought
        bought_profits
        # +
        # # sold
        # sold_profits
        # fee
        - (torch.abs(detached_y_test_pred) * TRADING_FEE)
    ) / detached_y_test_pred.numel()
    # todo random deprecation?
    return current_profit


def calculate_trading_profit_torch_with_buysell(scaler, last_values, y_test, y_test_pred,
                                                y_test_high, y_test_high_pred,
                                                y_test_low, y_test_low_pred):
    """
    Calculate trading profits
    :param y_test_low_pred:
    :param y_test_low:
    :param y_test_high_pred:
    :param y_test_high:
    :param last_values:
    :param y_test:
    :param y_test_pred:
    :return:
    """
    calculated_profit_values = calculate_trading_profit_torch_with_buysell_profit_values(y_test, y_test_high,
                                                                                         y_test_high_pred, y_test_low,
                                                                                         y_test_low_pred, y_test_pred)
    current_profit = torch.sum(
        # saved money
        # saved_money
        # +
        # bought
        calculated_profit_values
    )

    # todo random deprecation?
    return current_profit


def calculate_trading_profit_torch_with_buysell_profit_values(y_test, y_test_high, y_test_high_pred, y_test_low,
                                                              y_test_low_pred, y_test_pred):
    # reshape all args to 1d
    y_test = y_test.view(-1)
    y_test_pred = y_test_pred.view(-1)
    y_test_high = y_test_high.view(-1)
    y_test_high_pred = y_test_high_pred.view(-1)
    y_test_low = y_test_low.view(-1)
    y_test_low_pred = y_test_low_pred.view(-1)
    # make sure y_test_low_pred is lower than 0/reasonable
    y_test_low_pred = torch.clamp(y_test_low_pred, -1, 0)
    y_test_high_pred = torch.clamp(y_test_high_pred, 0, 10)
    percent_movements_scaled = y_test  # not scientific
    detached_y_test_pred = y_test_pred
    bought_profits = torch.clip(detached_y_test_pred, 0, 10) * percent_movements_scaled
    sold_profits = torch.clip(y_test_pred, -10, 0) * percent_movements_scaled
    # saved_money = torch.clamp(
    #             1 - torch.abs(y_test_pred), 0, 500
    #         )
    # / detached_y_test_pred.numel()
    # for the buys if we follow the y_test_high_pred to sell at, we can instead sell at the high only if its within that day
    # find points where y_test_high is greater than y_test_high_pred
    hit_high_points = y_test_high_pred * (y_test_high_pred <= y_test_high) * torch.clip(detached_y_test_pred, 0, 10)
    missed_high_points = bought_profits * (
            y_test_high_pred > y_test_high)  # already calculated/betted on but not cutoff
    # print("profit before hit_high_points: ", bought_profits)
    bought_adjusted_profits = hit_high_points + missed_high_points
    # print("profit hit_high_points: ", bought_adjusted_profits)
    # bought_adjusted_profits = torch.max(hit_high_points * (bought_adjusted_profits > 0).float(), bought_adjusted_profits)
    # find points where y_test_low is less than y_test_low_pred
    hit_low_points = (y_test_low_pred * (y_test_low_pred >= y_test_low)) * torch.clip(y_test_pred, -10, 0)
    missed_points = sold_profits * (
            y_test_low_pred < y_test_low)  # you are left with missed points/others already calculated
    # print("profit before hit_low_points: ", sold_profits)
    adjusted_profits = hit_low_points + missed_points
    # print("profit after hit_low_points: ", adjusted_profits)
    # sold_profits = torch.max(torch.abs(hit_low_points) * (sold_profits > 0).float(), sold_profits)
    calculated_profit_values = (bought_adjusted_profits
                                +
                                # sold
                                adjusted_profits
                                # fee
                                - (torch.abs(detached_y_test_pred) * TRADING_FEE))
    return calculated_profit_values


def calculate_trading_profit_torch_with_entry_buysell(scaler, last_values, y_test, y_test_pred,
                                                      y_test_high, y_test_high_pred,
                                                      y_test_low, y_test_low_pred):
    """
    Calculate trading profits
    :param y_test_low_pred:
    :param y_test_low:
    :param y_test_high_pred:
    :param y_test_high:
    :param last_values:
    :param y_test:
    :param y_test_pred:
    :return:
    """
    calculated_profit_values = calculate_profit_torch_with_entry_buysell_profit_values(y_test, y_test_high,
                                                                                       y_test_high_pred,
                                                                                       y_test_low, y_test_low_pred,
                                                                                       y_test_pred)

    current_profit = torch.sum(
        # saved money
        # saved_money
        # +
        # bought
        calculated_profit_values
    )

    # todo random deprecation?
    return current_profit


def calculate_profit_torch_with_entry_buysell_profit_values(y_test, y_test_high, y_test_high_pred, y_test_low,
                                                            y_test_low_pred,
                                                            y_test_pred):
    # reshape all args to 1d
    y_test = y_test.view(-1)
    y_test_pred = y_test_pred.view(-1)
    y_test_high = y_test_high.view(-1)
    y_test_high_pred = y_test_high_pred.view(-1)
    y_test_low = y_test_low.view(-1)
    y_test_low_pred = y_test_low_pred.view(-1)
    # make sure y_test_low_pred is lower than 0/reasonable
    y_test_low_pred = torch.clamp(y_test_low_pred, -1, 0)
    y_test_high_pred = torch.clamp(y_test_high_pred, 0, 10)
    percent_movements_scaled = y_test  # not scientific
    pred_low_to_close_percent_movements = torch.abs(y_test_low_pred - y_test)
    pred_low_to_high_percent_movements = torch.abs(y_test_low_pred - y_test_high_pred)
    pred_high_to_close_percent_movements = torch.abs(y_test_high_pred - y_test)
    pred_high_to_low_percent_movements = torch.abs(y_test_high_pred - y_test_low_pred)
    detached_y_test_pred = y_test_pred
    # bought profits are 0 if we don't hit the low or low to close
    bought_profits = torch.clip(detached_y_test_pred, 0, 10) * pred_low_to_close_percent_movements * (
            y_test_low_pred > y_test_low)  # miss out on buying if low is lower than low pred
    sold_profits = torch.clip(y_test_pred, -10, 0) * pred_high_to_close_percent_movements * (
            y_test_high_pred < y_test_high)  # miss out on selling if high is higher than high pred
    # saved_money = torch.clamp(
    #             1 - torch.abs(y_test_pred), 0, 500
    #         )
    # / detached_y_test_pred.numel()
    # for the buys if we follow the y_test_high_pred to sell at, we can instead sell at the high only if its within that day
    # find points where y_test_high is greater than y_test_high_pred
    hit_high_points = pred_low_to_high_percent_movements * (y_test_high_pred <= y_test_high) * torch.clip(
        detached_y_test_pred, 0, 10) * (
                              y_test_low_pred > y_test_low)  # miss out on buying if low is lower than low pred
    missed_high_points = bought_profits * (
            y_test_high_pred > y_test_high)  # already calculated/betted on but not cutoff
    # print("profit before hit_high_points: ", bought_profits)
    bought_adjusted_profits = hit_high_points + missed_high_points
    # print("profit hit_high_points: ", bought_adjusted_profits)
    # bought_adjusted_profits = torch.max(hit_high_points * (bought_adjusted_profits > 0).float(), bought_adjusted_profits)
    # find points where y_test_low is less than y_test_low_pred
    hit_low_points = -1 * pred_high_to_low_percent_movements * (y_test_low_pred >= y_test_low) * torch.clip(y_test_pred,
                                                                                                            -10, 0) * (
                             y_test_high_pred < y_test_high)  # miss out on selling if high is higher than high pred
    missed_points = sold_profits * (
            y_test_low_pred < y_test_low)  # you are left with missed points/others already calculated
    # print("profit before hit_low_points: ", sold_profits)
    adjusted_profits = hit_low_points + missed_points
    # print("profit after hit_low_points: ", adjusted_profits)
    # sold_profits = torch.max(torch.abs(hit_low_points) * (sold_profits > 0).float(), sold_profits)
    hit_trading_points = torch.logical_and((y_test_high_pred < y_test_high), (y_test_low_pred > y_test_low))
    calculated_profit_values = (bought_adjusted_profits +
                                adjusted_profits -
                                ((torch.abs(detached_y_test_pred) * TRADING_FEE) * hit_trading_points)  # fee
                                )
    return calculated_profit_values


# class TradingLossBinary(MultiHorizonMetric):
#     """
#     trading loss for use with pytorch forecasting
#     """
#
#     def loss(self, y_pred, target):
#         y_pred = self.to_prediction(y_pred)
#         loss = calculate_trading_profit_torch(None, None, target, (y_pred > 0).float() * 2 - 1)
#         return -loss


# class TradingLoss(MultiHorizonMetric):
#     """
#     trading loss for use with pytorch forecasting
#     """
#     _load_state_dict_post_hooks = OrderedDict()
#     def loss(self, y_pred, target):
#         y_pred = self.to_prediction(y_pred)
#         loss = calculate_trading_profit_torch(None, None, target, y_pred)
#         return -loss
#
#     # @property
#     # def _load_state_dict_post_hooks(self):
#     #     return {} # seems required now in new torch
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

    percent_movements_scaled = y_test  # not scientific
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


def percent_movements_augment(to_scale_tensor):
    """ scales a tensor so that the first element is baseline 1 and subsequent elements represent percentage change from the previous value"""
    arr = np.asarray(to_scale_tensor, dtype=float).flatten()
    values = [1.0]
    if arr.size > 1:
        from decimal import Decimal, ROUND_DOWN

        diffs = (arr[1:] - arr[:-1]) / arr[:-1]
        for val in diffs:
            values.append(float(Decimal(str(float(val))).quantize(Decimal("0.000"), rounding=ROUND_DOWN)))
    return values
    # return (to_scale_tensor - to_scale_tensor.shift(1)) / to_scale_tensor.shift(1)


def percent_movements_augment_to(to_scale_tensor, movement_to_tensor):
    """ percent changes moving from to scale to movement to"""
    output = (movement_to_tensor - to_scale_tensor) / to_scale_tensor
    return output
    # return (to_scale_tensor - to_scale_tensor.shift(1)) / to_scale_tensor.shift(1)


def calculate_takeprofit_torch(scaler, y_ndhalp_test, y_test, y_test_pred):
    """
    Calculate trading take profits
    :param y_ndhalp_train: how much the high is actually above this current close
    :param y_test: end close state
    :param y_test_pred: preds for how much more to sell at
    :return:
    """
    where_under = (y_test_pred < y_ndhalp_test).float()
    where_over = (y_test_pred >= y_ndhalp_test).float()
    under_sold_prices = y_test_pred * where_under
    hodl_prices = where_over * y_test
    current_profit = torch.sum(under_sold_prices)
    current_end_profits = torch.sum(hodl_prices)  # we predicted to sell higher so we get the real end value
    return (current_profit + current_end_profits) / y_test.numel()
    # percent_movements_scaled = y_test  # not scientific
    # detached_y_test_pred = y_test_pred
    # bought_profits = torch.clip(detached_y_test_pred, 0, 10) * percent_movements_scaled
    # sold_profits = torch.clip(y_test_pred, -10, 0) * percent_movements_scaled
    # # saved_money = torch.clamp(
    # #             1 - torch.abs(y_test_pred), 0, 500
    # #         )
    # current_profit = torch.sum(
    #     # saved money
    #     # saved_money
    #     # +
    #     # bought
    #     bought_profits
    #     +
    #     # sold
    #     sold_profits
    #     # fee
    #     - (torch.abs(detached_y_test_pred) * TRADING_FEE)
    # ) / len(detached_y_test_pred)
    # # todo random deprecation?
    # return current_profit


def calculate_takeprofit_torch_sq(scaler, y_ndhalp_test, y_test, y_test_pred):
    """
    Calculate trading take profits
    :param y_ndhalp_train: how much the high is actually above this current close
    :param y_test: end close state
    :param y_test_pred: preds for how much more to sell at
    :return:
    We also square things so its more sensitive to outliers avoiding soemthing?
    """
    where_under = (y_test_pred < y_ndhalp_test).float()
    where_over = (y_test_pred >= y_ndhalp_test).float()
    under_sold_prices = y_test_pred * where_under
    hodl_prices = where_over * y_test
    all_profits_sq = (under_sold_prices + hodl_prices)
    where_loosing = torch.sum(
        (all_profits_sq < 0).float()) / y_test.numel()  # we want to minimize loosing money add this weighted rate in
    # current_profit = torch.sum(under_sold_prices)
    # current_end_profits = torch.sum(hodl_prices)  # we predicted to sell higher so we get the real end value
    return (torch.sum(all_profits_sq) / y_test.numel()) * where_loosing  # mul by loosing rate
