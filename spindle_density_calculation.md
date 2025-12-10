# Spindle Density Calculation Per Region

## Overview

Spindle density is calculated as the number of spindles per minute for each net region (anterior vs posterior). This metric is used to analyze the relationship between sleep spindle activity and cognitive performance/OSA severity.

## Logic Flow

### 1. Channel Definitions
Regions are defined by electrode channels based on anatomical locations:

**Anterior channels** (E1-E69 + additional): frontal and central regions
**Posterior channels** (E70-E193): parietal and occipital regions

### 2. Data Collection
For each subject:
- Load detected spindles from `all_spindles_detailed.csv`
- Filter spindles by region channels
- Calculate recording duration from spindle end times

### 3. Density Calculation
```
spindle_density = total_spindles / (total_duration_seconds / 60)
```

## Code Implementation

### Channel Loading

```python
def load_anterior_channels(self):
    """Load anterior channel definitions from segment_net.txt"""
    anterior_channels_path = os.path.join(os.path.dirname(__file__), "reference", "segment_net.txt")

    with open(anterior_channels_path, 'r') as f:
        content = f.read()

    # Extract anterior channels section
    anterior_section = content.split('Anterior_channels:')[1].split('Posterior_channels:')[0]

    anterior_channels = []
    for line in anterior_section.strip().split('\n'):
        if line.strip():
            line_channels = [ch.strip() for ch in line.split(',') if ch.strip()]
            anterior_channels.extend(line_channels)

    # Convert to numeric values and sort
    anterior_numeric = []
    for channel in anterior_channels:
        if channel.startswith('E'):
            try:
                num = int(channel[1:])  # Remove 'E' prefix
                anterior_numeric.append(str(num))
            except ValueError:
                continue

    return sorted(list(set(anterior_numeric)))
```

### Density Calculation

```python
def collect_spindle_density_data(self, channels, region_name):
    """Collect spindle density data for specified channels."""

    # Find all spindle files
    pattern = os.path.join(self.project_dir, "derivatives", "sub-*",
                          "spindles", "results", "all_spindles_detailed.csv")
    spindle_files = glob.glob(pattern)

    spindle_data = []
    channels_int = [int(ch) for ch in channels]

    for file_path in spindle_files:
        try:
            # Extract subject ID
            path_parts = file_path.split(os.sep)
            subject_folder = [part for part in path_parts if part.startswith('sub-')][0]
            subject_id = subject_folder.replace("sub-", "")

            # Read spindle data
            spindle_df = pd.read_csv(file_path, encoding='utf-8')

            # Filter for specified channels
            region_spindles = spindle_df[spindle_df['Channel'].isin(channels_int)]

            if len(region_spindles) == 0:
                continue

            # Calculate spindle density
            total_duration_seconds = region_spindles['End'].max()
            total_duration_minutes = total_duration_seconds / 60
            total_spindles = len(region_spindles)
            spindle_density = total_spindles / total_duration_minutes if total_duration_minutes > 0 else 0

            spindle_data.append({
                'Subject': subject_id,
                f'{region_name.lower()}_spindle_density': spindle_density,
                f'{region_name.lower()}_total_spindles': total_spindles,
                f'{region_name.lower()}_recording_duration': total_duration_minutes
            })

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return pd.DataFrame(spindle_data)
```

### Key Calculation Details

1. **Duration Calculation**: Uses the maximum 'End' time from all spindles in the region as total recording duration
2. **Unit Conversion**: Converts seconds to minutes by dividing by 60
3. **Density Formula**: `spindles_per_minute = total_spindles / duration_minutes`
4. **Region Filtering**: Only includes spindles from channels assigned to each region

This provides spindle density in units of **spindles per minute** for each net region per subject.
