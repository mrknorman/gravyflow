
from gwpy.timeseries import TimeSeries
import logging
import time

logging.basicConfig(level=logging.INFO)

print("Checking O1 Data Availability...")

# GW150914 GPS: 1126259462
start = 1126259460
end = 1126259464
ifo = "H1"

# Try C01 (Current Config)
print(f"Attempt 1: C01 (H1_DCS-CALIB_STRAIN_CLEAN_C01)")
try:
    start_time = time.time()
    # Note: gwpy might try to find files. 
    # find_urls signature is what acquisition uses.
    from gwpy.io.locator import find_urls
    files = find_urls(
        site="H",
        frametype="H1_DCS-CALIB_STRAIN_CLEAN_C01",
        gpsstart=start,
        gpsend=end,
        urltype="file"
    )
    print(f"Found {len(files)} files in {time.time()-start_time:.2f}s")
except Exception as e:
    print(f"Failed: {e}")

# Try Public (H1_LOSC_4_V1)
print(f"Attempt 2: Public (H1_LOSC_4_V1)")
try:
    start_time = time.time()
    # For public data, urltype might not be file? 
    # But usually One needs to fetch availability.
    # Let's try TimeSeries.fetch_open_data
    ts = TimeSeries.fetch_open_data(ifo, start, end)
    print(f"Success! Fetched public data in {time.time()-start_time:.2f}s")
except Exception as e:
    print(f"Failed public fetch: {e}")
