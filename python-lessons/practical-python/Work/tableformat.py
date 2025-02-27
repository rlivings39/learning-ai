'''
Table formatting tools
'''

# tableformat.py

class TableFormatter:
    def __init__(self):
        self._report_str = ''

    def headings(self, headers):
        '''
        Emit the table headings.
        '''
        raise NotImplementedError()

    def row(self, rowdata):
        '''
        Emit a single row of table data.
        '''
        raise NotImplementedError()

    def __str__(self):
        return self._report_str

class TextTableFormatter(TableFormatter):
    '''
    Emit a table in plain-text format
    '''
    def __init__(self):
        super().__init__()
        self._width = 10

    def headings(self, headers):
        num_cols = len(headers)
        self._report_str = (f'{{:>{self._width}s}} ' * num_cols).format(*headers) + '\n'
        self._report_str += ('-'*self._width + ' ') * num_cols + "\n"

    def row(self, rowdata):
        element_fmt = f'{{:>{self._width}s}} '*len(rowdata) + "\n"
        self._report_str += element_fmt.format(*rowdata)

class CSVTableFormatter(TableFormatter):
    '''
    Output data in CSV format
    '''
    def headings(self, headers):
        self._report_str += ','.join(headers) + "\n"

    def row(self, rowdata):
        self._report_str += ','.join(rowdata) + "\n"

class HTMLTableFormatter(TableFormatter):
    '''
    Output data in HTML format
    '''
    def _format_row(self, rowdata, is_header: bool):
        element_tag = 'th' if is_header else 'td'
        col_fmt = f'<{element_tag}>{{:s}}</{element_tag}>'*len(rowdata)
        self._report_str += '<tr>' + col_fmt.format(*rowdata) + '</tr>\n'

    def headings(self, headers):
        self._format_row(headers, is_header=True)

    def row(self, rowdata):
        self._format_row(rowdata, is_header=False)

def create_formatter(fmt: str):
    '''
    Create a formatter of the right type based on format arg which can be "txt", "csv", "html"
    '''
    formatter = None
    if fmt == "txt":
        formatter = TextTableFormatter()
    elif fmt == "csv":
        formatter = CSVTableFormatter()
    elif fmt == "html":
        formatter = HTMLTableFormatter()
    else:
        raise ValueError(f'Unknown format value {fmt}')
    return formatter
