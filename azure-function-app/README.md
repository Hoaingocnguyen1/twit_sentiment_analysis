# Azure Function App

This project contains an Azure Function that is triggered by new files added to Azure Blob Storage. The function reads the contents of the file and inserts the data into a PostgreSQL database.

## Project Structure

```
azure-function-app
├── BlobTriggerFunction
│   ├── __init__.py        # Main function logic
│   ├── function.json      # Function configuration
│   └── requirements.txt    # Python dependencies
├── local.settings.json     # Local configuration settings
├── host.json               # Global configuration options
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd azure-function-app
   ```

2. **Install dependencies**:
   Navigate to the `BlobTriggerFunction` directory and install the required packages:
   ```
   cd BlobTriggerFunction
   pip install -r requirements.txt
   ```

3. **Configure local settings**:
   Update the `local.settings.json` file with your Azure Blob Storage and PostgreSQL connection details.

4. **Run the function locally**:
   Use the Azure Functions Core Tools to run the function locally:
   ```
   func start
   ```

## Usage

- The function will automatically trigger when a new file is uploaded to the specified Blob Storage container.
- Ensure that the file format is compatible with the processing logic defined in `__init__.py`.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.
