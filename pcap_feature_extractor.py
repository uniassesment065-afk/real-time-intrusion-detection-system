import pandas as pd
import numpy as np

def extract_features_from_pcap(pcap_file, num_packets=50):
    """
    Dummy feature extractor simulating network traffic.
    Returns a DataFrame with 42 features per packet.
    """
    
    # Simulate packet sizes (bytes)
    packet_sizes = np.random.randint(20, 1500, size=num_packets)  # typical packet sizes
    
    # Simulate inter-arrival times (ms)
    inter_arrival = np.random.exponential(scale=50, size=num_packets)  # average ~50ms
    
    # Simulate protocols (0=TCP, 1=UDP, 2=ICMP)
    protocols = np.random.choice([0, 1, 2], size=num_packets)
    
    # Simulate some random statistics per packet (for remaining features)
    extra_features = np.random.rand(num_packets, 39)  # 42-3 = 39 remaining features
    
    # Combine all features into a single array
    features = np.column_stack((packet_sizes, inter_arrival, protocols, extra_features))
    
    # Build DataFrame with feature names f0..f41
    df = pd.DataFrame(features, columns=[f"f{i}" for i in range(42)])
    
    return df