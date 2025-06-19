import os

def error_message_detail(error,error_detail):
    _,_,exec_tb=error_detail.exc_info()
    if exec_tb is not None:
        file_name=os.path.split(exec_tb.tb_frame.f_code.co_filename)[1]
        return f"Error occured python script name [{0} line number [{1}] error message [{2}]]".format(
            file_name,exec_tb.tb_lineno, str(error)
            )
    else:
        return f"Error message:{str(error)}"

class CustomException(Exception):
    def __init__(self,error_message,error_detail):
        """
        param : error_message : error message in string format
        """
        super().__init__(error_message)
        self.error_message=error_message_detail(
            error_message,error_detail=error_detail
        )

    def __str__(self):
        return self.error_message