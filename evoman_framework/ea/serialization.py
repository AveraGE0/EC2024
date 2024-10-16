"""Module for serializing outcomes of the ea runs."""
import os
import pickle


def save_best_individual(
        experiment_path: str,
        population: list[list],
        metric: str,
        best="highest"
    ) -> None:
    """Function to extract the best individual and to save it to a file.
    Additionally, the weights of the individual are saved as txt, to
    have them ready for submission.

    Args:
        population (list[list]): Population from which best individual is extracted.
        metric (str): Metric used to determine the best individual.
        best (str, optional): Indicator if the best value of the metric is the 'highest'
                              or 'lowest' value. Options: "highest", "lowest".
                              Defaults to "highest".
    """
    reverse = {  # sorting is ascending, so with highest we reverse the sorting order.
        "highest": True,
        "lowest": False
    }[best]

    best_individual = sorted(
        population,
        reverse=reverse,
        key=lambda x: get_nested_attribute(x, metric)
    )[0]

    print(
        f"Saving best individual (metric={metric}) with value of="\
        f"{round(get_nested_attribute(best_individual, metric), 3)}"
    )

    with open(os.path.join(experiment_path, f'best_individual_{metric}.pkl'), 'wb') as i_file:
        pickle.dump(best_individual, i_file)


def get_nested_attribute(obj: object, attribute: str) -> any:
    """Function to access nested and indexed attributes of objects.
    For example, fitness.values[0] given as a string can be accessed
    using this function.

    Args:
        obj (object): Object from which the attribute should be retrieved.
        attribute (str): (Nested) attribute that should be accessed on the object.

    Returns:
        any: The value at the attribute of the object.
    """
    attributes = attribute.split(".")

    for attr in attributes:
        if "[" in attr:
            ind_start = attr.index("[")
            ind_end = attr.index("]")

            indexer = int(attr[ind_start+1:ind_end])
            obj = getattr(obj, attr[:ind_start])[indexer]
        else:
            obj = getattr(obj, attr)

    return obj
