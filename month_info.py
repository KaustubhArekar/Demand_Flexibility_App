

def month_info(year):
    import calendar
    """
    Get information for all months in a given year.
    """
    months_info = {}
    cumulative_days = 1
    
    for month_num in range(1, 13):
        month_name = calendar.month_name[month_num]
        num_days = calendar.monthrange(year, month_num)[1]
        
        months_info[month_name.lower()] = {
            'month_num': month_num,
            'num_days': num_days,
            'start_day': cumulative_days,
            'end_day': cumulative_days + num_days - 1,
            'range': f"{cumulative_days} - {cumulative_days + num_days - 1}"
        }
        
        cumulative_days += num_days
    
    return months_info
