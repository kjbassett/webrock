import unittest
from webrock.app import prepare_form_data


class Test_prepare_form_data(unittest.TestCase):

    def test_prepare_form_data_raises_value_error_for_missing_required_argument(self):
        meta = {
            "args": [
                {"name": "required_arg", "type": "str"}
            ]
        }
        form = {}

        with self.assertRaises(ValueError) as context:
            prepare_form_data(meta, form)

        self.assertEqual(str(context.exception), "Missing required argument required_arg")

    def test_prepare_form_data_uses_default_value_when_argument_not_in_form(self):
        meta = {
            "args": [
                {"name": "optional_arg", "type": "str", "default": "default_value"}
            ]
        }
        form = {}

        result = prepare_form_data(meta, form)

        self.assertEqual(result["optional_arg"], "default_value")

    def test_prepare_form_data_converts_form_values_to_specified_type(self):
        meta = {
            "args": [
                {"name": "int_arg", "type": "int"},
                {"name": "float_arg", "type": "float"},
                {"name": "str_arg", "type": "str"},
                {"name": "bool_arg_true", "type": "bool"},
                {"name": "bool_arg_false", "type": "bool"}
            ]
        }
        form = {
            "int_arg": ["42"],
            "float_arg": ["3.14"],
            "str_arg": ["hello"],
            "bool_arg_true": ["True"],
            "bool_arg_false": ["False"]
        }

        result = prepare_form_data(meta, form)

        self.assertEqual(result["int_arg"], 42)
        self.assertEqual(result["float_arg"], 3.14)
        self.assertEqual(result["str_arg"], "hello")
        self.assertEqual(result["bool_arg_true"], True)
        self.assertEqual(result["bool_arg_false"], False)

    def test_prepare_form_data_raises_error_for_invalid_type(self):
        meta = {
            "args": [
                {"name": "invalid_type_arg", "type": "nonexistent_type"}
            ]
        }
        form = {
            "invalid_type_arg": ["some_value"]
        }

        with self.assertRaises(ValueError) as context:
            prepare_form_data(meta, form)

        self.assertIn("nonexistent_type", str(context.exception))

    def test_prepare_form_data_processes_empty_form_with_defaults(self):
        meta = {
            "args": [
                {"name": "arg1", "type": "str", "default": "default1"},
                {"name": "arg2", "type": "int", "default": 10},
                {"name": "arg3", "type": "float", "default": 3.14},
                {"name": "arg4", "type": "bool", "default": True}
            ]
        }
        form = {}

        result = prepare_form_data(meta, form)

        self.assertEqual(result["arg1"], "default1")
        self.assertEqual(result["arg2"], 10)
        self.assertEqual(result["arg3"], 3.14)
        self.assertEqual(result["arg4"], True)

    def test_prepare_form_data_handles_form_with_all_arguments_present(self):
        meta = {
            "args": [
                {"name": "arg1", "type": "str"},
                {"name": "arg2", "type": "int"},
                {"name": "arg3", "type": "float"},
                {"name": "arg4", "type": "bool"}
            ]
        }
        form = {
            "arg1": ["value1"],
            "arg2": ["123"],
            "arg3": ["45.67"],
            "arg4": ["True"]
        }

        result = prepare_form_data(meta, form)

        self.assertEqual(result["arg1"], "value1")
        self.assertEqual(result["arg2"], 123)
        self.assertEqual(result["arg3"], 45.67)
        self.assertEqual(result["arg4"], True)

    def test_prepare_form_data_raises_error_for_non_convertible_value(self):
        meta = {
            "args": [
                {"name": "int_arg", "type": "int"}
            ]
        }
        form = {
            "int_arg": ["not_an_int"]
        }

        with self.assertRaises(ValueError) as context:
            prepare_form_data(meta, form)

        self.assertIn("invalid literal for int()", str(context.exception))

    def test_prepare_form_data_prefers_specified_values_over_default(self):
        meta = {
            "args": [
                {"name": "opt_arg1", "type": "str", "default": "default1"},
                {"name": "opt_arg2", "type": "int", "default": 10},
                {"name": "opt_arg3", "type": "float", "default": 3.14},
                {"name": "opt_arg4", "type": "bool", "default": False}
            ]
        }
        form = {
            "opt_arg1": ["provided_value1"],
            "opt_arg2": ["20"],
            "opt_arg3": ["6.28"],
            "opt_arg4": ["True"]
        }

        result = prepare_form_data(meta, form)

        self.assertEqual(result["opt_arg1"], "provided_value1")
        self.assertEqual(result["opt_arg2"], 20)
        self.assertEqual(result["opt_arg3"], 6.28)
        self.assertEqual(result["opt_arg4"], True)

    def test_prepare_form_data_does_not_convert_type_any(self):
        meta = {
            "args": [
                {"name": "any_arg", "type": "any"}
            ]
        }
        form = {
            "any_arg": ["some_value"]
        }

        result = prepare_form_data(meta, form)

        self.assertEqual("some_value", result["any_arg"])
