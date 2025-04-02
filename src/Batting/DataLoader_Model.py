

import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class FantasyData(Dataset):
    def __init__(self,device,df):
        x,y= self.preprocess_batting(df)
        self.x = x.to(device)
        self.y = y.to(device)
        self.device = device
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def transform_batting_data(self , df):

        self.batter_data = {}
        # Ensure the data is sorted by season and match_id
        df = df[df['season'] > 2015]
        # df = df.sort_values(["season", "match_id"]).reset_index(drop=True)

        # Get all unique venues for one-hot encoding later
        # unique_venues = sorted(df["venue"].unique())

        # List to store each row's transformed dictionary
        transformed_rows = []

        # Process each batter separately
        for batter, group in df.groupby("fullName"):
            prev = -1
            # Sort the batter's matches in chronological order
            group = group.sort_values(["season", "match_id"]).reset_index(drop=True)
            # List to keep track of previous match scores for this batter
            previous_scores = []

            # Iterate over the batter's matches
            for i, row in group.iterrows():
                current_season = row["season"]

                # Previous match score: last score in the list, if available
                prev_match_score = previous_scores[-1] if previous_scores else None

                # Previous 5 matches average points: average of last 5 scores
                last_five = previous_scores[-5:] if previous_scores else []
                prev_5_avg = sum(last_five) / len(last_five) if last_five else None

                # Previous season average points: consider all matches of this batter with season < current_season
                prev_season_scores = group[group["season"] < current_season]["Batting_FP"].tolist()
                prev_season_avg = sum(prev_season_scores) / len(prev_season_scores) if prev_season_scores else None

                # Number of matches played so far (i.e. count of previous scores)
                num_matches = len(previous_scores)

                # One hot encode the venue for this match
                # venue_encoding = {v: 1 if row["venue"] == v else 0 for v in unique_venues}

                # Create the dictionary for the current row
                row_dict = {
                    "Batter name": row["fullName"],
                    "prev": prev_match_score,
                    "prev5": prev_5_avg,
                    "prevSSN": prev_season_avg,
                    "num matches": num_matches,
                    "venue": row["venue"],
                    "season": current_season,
                    "Batting_FP" : row["Batting_FP"],
                }
                # row_dict.extend(venue_encoding)

                transformed_rows.append(row_dict)
                prev = row_dict
                # Update the list of previous scores with the current match's score
                previous_scores.append(row["Batting_FP"])
            self.batter_data[batter] = prev

        return transformed_rows


    def get_batting_data(self,name,venue , season ):
        data = self.batter_data[name].copy()
        data["venue"] = [venue]
        data["season"] = [season]
        data = pd.DataFrame(data)
        data["Batting_FP"] = [0]*len(data)
        return self.preprocess_batting(data,False)

    def preprocess_batting(self, df , train = True  ):
        if train:
            df  = pd.DataFrame(self.transform_batting_data(df))
            self.temp = df


        # Targets

        y =  torch.tensor(
            df[["Batting_FP"]].values,
            dtype=torch.float32
        )

        # Compute sums for each column
        positive_sums = torch.sum(y * (y > 0), dim=0)  # Sum of positive values
        negative_sums = torch.sum(-y * (y < 0), dim=0)  # Sum of negative values

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        scale_factors = positive_sums / (negative_sums + epsilon)
        scale_factors = scale_factors.unsqueeze(0) * 0.05  # Shape: (1, 4)

        # Apply scaling only to negative values (correct broadcasting)
        y = torch.where(y < 0, y * scale_factors, y)
        print(y.shape)






        # 2. One-hot encode teams
        if train :
            self.venue_enc = OneHotEncoder(sparse_output=False)
            venue = self.venue_enc.fit_transform(df[['venue']])
            print(self.venue_enc)
        else :

            print([[df['venue']]])
            venue = self.venue_enc.transform(df[['venue']])
            print(venue)


        # 3. Process season
        if train :
            self.scaler = MinMaxScaler()
            season_scaled = self.scaler.fit_transform(df[['season']])
        else :
            season_scaled = self.scaler.transform(df[['season']])

        # Convert to tensors and combine
        season_tensor = torch.tensor(season_scaled, dtype=torch.float32)
        venue_tensor = torch.tensor(venue, dtype=torch.float32)

        prev = torch.tensor(df[["prev"]].fillna(0).values, dtype=torch.float32)
        prev5 = torch.tensor(df[["prev5"]].fillna(0).values, dtype=torch.float32)
        prevSSN = torch.tensor(df[["prevSSN"]].fillna(0).values, dtype=torch.float32)
        num_matches = torch.tensor(df[["num matches"]].fillna(0).values, dtype=torch.float32)

        x = torch.cat([
            prev,
            prev5,
            prevSSN,
            num_matches,
            season_tensor,
            venue_tensor,
        ], dim=1)
        # print(x.shape)

        return x,y
class BattingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(28,128,bias=True)
        self.lin2 = nn.Linear(128,128,bias=True)
        self.lin3 = nn.Linear(128,64,bias=True)
        self.lin4 = nn.Linear(64,32,bias=True)
        self.lin5 = nn.Linear(32,1,bias=True)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.relu(x)
        x = self.lin5(x)
        # x = self.relu(x)
        return x
