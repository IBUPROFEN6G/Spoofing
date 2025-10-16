
import uuid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sortedcontainers import SortedList

class Order:
    def __init__(self, price, size):
        self.ID = str(uuid.uuid4())  # order id
        self.price = price  # order price
        self.size = size  # order size when placing
        self.remaining_size = size  # the standing size in LOB

    def __str__(self):
        return f"ID: {self.ID}\nPrice: {self.price}\nSize: {self.size}\nRemaining Size: {self.remaining_size}"

    def __repr__(self):
        return f"ID: {self.ID} | Price: {self.price} | Size: {self.size} | Remaining Size: {self.remaining_size}"

class One_side_queue:
    def __init__(self, direction):
        self.direction = direction.lower()
        if direction == "bid":
            self.queue = SortedList([], key=lambda order: -order.price)
        elif direction == "ask":
            self.queue = SortedList([], key=lambda order: order.price)
        self.order_dict = {}

    def add(self, *args):
        # add single order
        if len(args) == 1 and isinstance(args[0], Order):
            self.queue.add(args[0])
        # add a list of orders
        elif len(args) == 1 and isinstance(args[0], list):
            self.queue.update(args[0])
        else:
            raise InputError("Add: add an order or list of orders.")

    def remove(self, order_id):
        # remove by id
        if order_id in self.order_dict:
            order = self.order_dict[order_id]
            self.queue.remove(order)
            del self.order_dict[order_id]
        # remove by index
        elif isinstance(order_id, int):
            # remove
            if index is not None:
                self.queue.pop(index)
            else:
                raise OrderNotFoundError

    def remove_empty(self):
          empty_orders = [order for order in self.queue if order.remaining_size == 0]
          for order in empty_orders:
            self.queue.remove(order)
            if order.ID in self.order_dict:
                del self.order_dict[order.ID]

    def inquire(self, ID):
        return any(cur_order.ID == ID for cur_order in self.queue)

    def inquire_remaining_size(self, ID):
        for cur_order in self.queue:
            if cur_order.ID == ID:
                return cur_order.remaining_size

    def __getitem__(self, index):
        return self.queue[index]

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        output = ""
        # header
        if self.direction == "bid":
            output += "      BID SIDE      \n"
        elif self.direction == "ask":
            output += "      ASK SIDE      \n"
        output += " QUANTITY    PRICE  \n"
        # order
        for cur_order in self.queue:
            output += f"  [{cur_order.size:.2f}  -  {cur_order.price:.2f}] \n"

        return output

    def __repr__(self):
        return self.__str__()

    class LOB:
      def __init__(self):
        self.bid_side = One_side_queue(direction="bid")
        self.ask_side = One_side_queue(direction="ask")

      def get_best_bid_price(self):
        if len(self.bid_side) == 0:
            raise RunOutOfOrderError('bid')

        return self.bid_side[0].price

      def get_best_ask_price(self):
        if len(self.ask_side) == 0:
            raise RunOutOfOrderError('ask')

        return self.ask_side[0].price

      def get_mid_price(self):
        return (self.get_best_bid_price() + self.get_best_ask_price()) / 2

      def get_num_orders(self, direction):
        if direction.lower() == "bid":
            return len(self.bid_side)
        elif direction.lower() == "ask":
            return len(self.ask_side)
        else:
            raise SideKeyError

      def get_volume(self, direction):
        if direction.lower() == "bid":
            return sum(cur_order.remaining_size for cur_order in self.bid_side)
        elif direction.lower() == "ask":
            return sum(cur_order.remaining_size for cur_order in self.ask_side)
        else:
            raise SideKeyError

      def add(self, *args):
        if args[0].lower() == "bid":
            self.bid_side.add(args[1])
        elif args[0].lower() == "ask":
            self.ask_side.add(args[1])
        else:
            raise SideKeyError

      def remove(self, *args):
        if args[0].lower() == "bid":
            self.bid_side.remove(args[1])
        elif args[0].lower() == "ask":
            self.ask_side.remove(args[1])
        else:
            raise SideKeyError

      def inquire(self, direction, ID):
        if direction.lower() == "bid":
            found = self.bid_side.inquire(ID)
        elif direction.lower() == "ask":
            found = self.ask_side.inquire(ID)
        else:
            raise SideKeyError

        return found



    def matching(self):
      while self.bid_side and self.ask_side:
        best_bid = self.bid_side[0]
        best_ask = self.ask_side[0]

        if best_bid.price < best_ask.price:
            break


        trade_quantity = min(best_bid.remaining_size, best_ask.remaining_size)


        best_bid.remaining_size -= trade_quantity
        best_ask.remaining_size -= trade_quantity


        if best_bid.remaining_size == 0:
            self.bid_side.remove(best_bid.ID)
        if best_ask.remaining_size == 0:
            self.ask_side.remove(best_ask.ID)

    def random_generate(self, num_orders=10, spread=0.02, volatility=0.1):
      if len(self.bid_side) == 0 or len(self.ask_side) == 0:
        mid_price = 100.0
      else:
        mid_price = self.get_mid_price()


      bid_prices = mid_price * (1 - spread/2) * (1 + np.random.normal(0, volatility, num_orders))
      ask_prices = mid_price * (1 + spread/2) * (1 + np.random.normal(0, volatility, num_orders))


      bid_prices = np.clip(bid_prices, None, mid_price * 0.99)
      ask_prices = np.clip(ask_prices, mid_price * 1.01, None)

    def price_level_dict(self, level=0):
        # bid side
        bid_level_dist = {}
        for cur_order in self.bid_side:
            cur_price = round(cur_order.price, level)
            cur_size = cur_order.remaining_size
            if cur_price in bid_level_dist:
                bid_level_dist[cur_price] += cur_size
            else:
                bid_level_dist[cur_price] = cur_size

        # ask size
        ask_level_dist = {}
        for cur_order in self.ask_side:
            cur_price = round(cur_order.price, level)
            cur_size = cur_order.remaining_size
            if cur_price in ask_level_dist:
                ask_level_dist[cur_price] += cur_size
            else:
                ask_level_dist[cur_price] = cur_size

        return bid_level_dist, ask_level_dist

    def __str__(self):
        output = "           LIMIT ORDER BOOK            \n"
        output += "      BID SIDE            ASK SIDE     \n"
        output += " QUANTITY    PRICE   PRICE    QUANTITY \n"

        len_difference = abs(len(self.ask_side) - len(self.bid_side))
        len_min = min(len(self.ask_side), len(self.bid_side))
        longer_side = "bid" if len(self.bid_side) > len(self.ask_side) else "ask"

        for i in range(len_min):
            output += f"  [{self.bid_side[i].remaining_size:.2f}  -  {self.bid_side[i].price:.2f}] | [{self.ask_side[i].price:.2f}  -  {self.ask_side[i].remaining_size:.2f}] \n"

        for j in range(len_min, len_min + len_difference):
            if longer_side == "bid":
                output += (
                    f"  [{self.bid_side[j].remaining_size:.2f}  -  {self.bid_side[j].price:.2f}] |"
                    + " " * 19
                    + "\n"
                )
            else:
                output += (
                    " " * 19
                    + f"| [{self.ask_side[j].price:.2f}  -  {self.ask_side[j].remaining_size:.2f}] \n"
                )

        return output

    def __repr__(self):
        return self.__str__()


# Exceptions
    class OrderBookError(Exception):
      pass

    class SideKeyError(OrderBookError):
      def __str__(self):
        return """Input side is neither bid nor ask."""

    class PartialFillError(OrderBookError):
      def __str__(self):
        return """Some orders are not in LOB."""

    class OrderNotFoundError(OrderBookError):
      def __init__(self,order_id):
        self.order_id = order_id

      def __str__(self):
        return f"Order {self.order_id} not in LOB."

    class RunOutOfOrderError(OrderBookError):
      def __init__(self, side):
        self.side = side

    def __str__(self):
        return f"Run out of order on {self.side}."
    class InputError(OrderBookError):
      def __init__(self, message="Invalid input"):
        super(InputError, self).__init__()
        self.message = message

    def __str__(self):
        return f"{self.message}"



    def plot_book_directly(bid_levels, ask_levels):

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


      if bid_levels:
        prices = list(bid_levels.keys())
        volumes = list(bid_levels.values())
        ax1.barh(prices, volumes, color='green', alpha=0.7)
        ax1.set_title('Bid Side')


      if ask_levels:
        prices = list(ask_levels.keys())
        volumes = list(ask_levels.values())
        ax2.barh(prices, volumes, color='red', alpha=0.7)
        ax2.set_title('Ask Side')

      plt.show()
