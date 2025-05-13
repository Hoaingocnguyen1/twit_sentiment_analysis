from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

# from plugins.scripts.crawling_tweet import crawl_task
from plugins.scripts.crawling_tweet import crawl_task


class CrawlTweetsOperator(BaseOperator):
    """
    Custom Operator to crawl tweets and save them to a specified file.
    """

    template_fields = ["output_file"]  # Khai báo output_file là template field

    @apply_defaults
    def __init__(
        self, base_dir: str, hashtags_dict: dict, output_file: str, *args, **kwargs
    ):
        """
        :param base_dir: Directory to save the crawled tweets.
        :param hashtags_dict: Dictionary of hashtags grouped by topics.
        :param output_file: Path to the output file where tweets will be saved.
        """
        super().__init__(*args, **kwargs)
        self.base_dir = base_dir
        self.hashtags_dict = hashtags_dict
        self.output_file = output_file

    def execute(self, context):
        output_file = self.output_file
        # Log thông tin về file đầu ra
        self.log.info(f"Starting tweet crawling task. Output file: {self.output_file}")
        self.log.info(f"Base directory: {self.base_dir}")
        self.log.info(f"Hashtags dictionary: {self.hashtags_dict}")

        # Gọi hàm crawl_task để thực hiện crawling
        try:
            crawl_task(
                base_dir=self.base_dir,
                hashtags_dict=self.hashtags_dict,
                output_file=output_file,
                **context,  # Truyền context để hỗ trợ các thông tin bổ sung
            )
            self.log.info("Tweet crawling task completed successfully.")
        except Exception as e:
            self.log.error(f"Error during tweet crawling task: {e}")
            raise
