
from multiprocessing import Queue, Process

crypto_symbol_to_orders = {

}

queue = Queue()

def run_gpu_pool(input_queue):
    # setup
    while True:
        try:
            # get latest data for each crypto symbols
            latest_data(symbol)



            # block until something is placed on the queue
            # uid, symbol, side, price = input_queue.get()
            try:
                # poll untill
            except Exception as e:
                # We never want to crash this process
                logging.error(e)
                import traceback

                traceback.print_exc()
                traceback.print_tb(e.__traceback__)
                # failing tasks are always passed back
                output_queue.put((uid, e))
                continue
            output_queue.put((uid, alpha))
        except Exception as e:
            logging.error(e)
            pass

p = Process(
            target=run_gpu_pool,
            args=(
                queue,
                output_gpu_queue,
                "checkpoints/gca-dist-all-data/latest_model.pth",
                networks,
                CONFIG,
                torch,
                utils,
            ),
        )
        p.start()
