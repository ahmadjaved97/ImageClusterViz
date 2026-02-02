"""
Grouping algorithm for duplicate detection.
"""

from typing import Dict, List, Tuple, Set
from ..base import GroupingAlgorithm

class GroupBuilder(GroupingAlgorithm):
    """
    Grouping duplicate pairs into clusters using simple dictionary-based approach.
    """

    def group_duplicates(
        self,
        pairs,
        filenames
    ):
        """
        Group duplicate pairs into clusters.
        """

        # Tracking which group each image belongs to
        image_to_group ={}
        # Track members of each group
        groups = {}
        group_counter = 0

        # Process all pairs
        for img1, duplicates in paris.items():
            for img2, score in duplicates:
                # Get or create groups for both images.
                group1 = image_to_group.get(img1)
                group2 = image_to_group.get(img2)

                if group1 is None and group2 is None:
                    #Neither in a group - create new group
                    new_group_id = group_counter
                    group_counter += 1

                    groups[new_group_id] = {img1, img2}
                    image_to_group[img1] = new_group_id
                    image_to_group[img2] = new_group_id
                
                elif group1 is None:
                    # Only img2 has group - add img1 to it
                    groups[group2].add(img1)
                    image_to_group[img1] = group2
                
                elif group2 is None:
                    # Only img1 has group - add img2 to it
                    groups[group1].add(img2)
                    image_to_group[img2] = group1
                
                elif group1 != group2:
                    # Both in different groups - merge them
                    # Merge group2 into group1
                    groups[group1].update(groups[group2])

                    # Update all members of group2 to point to group1
                    for img in groups[group2]:
                        image_to_group[img] = group1
                    
                    # Remove old group
                    del groups[group2]
        

        # Convert sets to lists and return
        #Group ID is not meaningful, so we'll use first image as key
        result = {}
        for group_id, members in groups.items():
            members_list = sorted(list(members))
            # Fist member becomes the representative
            representative = members_list[0]
            # Rest are duplicates
            duplicates = members_list[1:]

            if duplicates:  # Only include if there are actual duplicates.
                result[representative] = duplicates
    
        
        return result

class PairwiseGrouping(GroupingAlgorithm):
    """
    No grouping - just return the pair as-is.
    """

    def group_duplicates(
        self,
        pairs,
        filenames
    ):
        """
        Returns pairs without grouping.
        """

        # Convert to group format.
        # Each image with duplicates becomes a "group".

        result = {}

        for img, duplicates in pairs.items():
            if duplicates:
                # Extract just the filenames (remove scores)
                dup_filenames = [dup[0] for dup in duplicates]
                result[img] = dup_filenames
        
        return result