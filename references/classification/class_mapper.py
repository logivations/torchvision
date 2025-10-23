#  (C) Copyright
#  Logivations GmbH, Munich 2010-2023
class ClassMapper:
    def __init__(self, num_classes: int = 4):
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        # Generate thresholds by dividing [0, 1] into bins
        self.thresholds = [(i + 0.7) / num_classes for i in range(num_classes - 1)]

    def handling_units_from_output(self, pred_value: float, orientation_value: float = None) -> int:
        """
        Maps prediction value (0–1) into discrete class index.
        If orientation is provided, uses it to refine ambiguous predictions.

        :param pred_value: Handling units prediction [0, 1]
        :param orientation_value: Orientation prediction [0, 1] (optional)
        :return: Discrete handling units count
        """

        basic_pred = self._basic_prediction(pred_value)

        if orientation_value is None:
            return basic_pred

        return self._refine_with_orientation(pred_value, orientation_value, basic_pred)

    def _basic_prediction(self, pred_value: float) -> int:
        """Basic threshold-based prediction"""
        for i, t in enumerate(self.thresholds):
            if pred_value < t:
                return i
        return self.num_classes - 1

    def _refine_with_orientation(self, pred_value: float, orientation_value: float, basic_pred: int) -> int:
        """
        Refines prediction using orientation for ambiguous cases.

        Logic:
        - If basic_pred is 0 or 1: return as-is
        - If basic_pred is 2 or 3 and pred_value is in ambiguous zone (around 2):
          * orientation >= 0.5 (lengthwise): cap at 2
          * orientation < 0.5 (widthwise): allow 3
        """
        actual_value = pred_value * (self.num_classes - 1)

        if basic_pred <= 1 or actual_value >= 2.5:
            return basic_pred

        if 1.5 <= actual_value < 2.5:
            if orientation_value >= 0.5:
                # Lengthwise: maximum 2 pallets
                return min(basic_pred, 2)
            else:
                # Widthwise: can be 3 pallets
                return basic_pred

        return basic_pred