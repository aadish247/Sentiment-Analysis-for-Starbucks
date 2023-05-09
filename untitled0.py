# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 00:07:50 2023

@author: aadis
"""

# List of 100 most popular Starbucks beverages
starbucks_beverages = [
    'Caffè Latte', 'Cappuccino', 'Caffè Mocha', 'Espresso', 'Americano', 'Vanilla Latte', 
    'Caramel Macchiato', 'Flat White', 'White Chocolate Mocha', 'Iced Coffee', 'Iced Caffè Americano', 
    'Iced Latte', 'Iced Caramel Macchiato', 'Iced White Chocolate Mocha', 'Frappuccino (various flavors)', 
    'Green Tea Latte', 'Chai Latte', 'Hot Chocolate', 'Steamed Milk', 'Skinny Vanilla Latte', 
    'Cinnamon Dolce Latte', 'Mocha Frappuccino', 'Java Chip Frappuccino', 'White Mocha Frappuccino', 
    'Caramel Frappuccino', 'Caramel Brulée Latte', 'Pumpkin Spice Latte', 'Eggnog Latte', 
    'Peppermint Mocha', 'Gingerbread Latte', 'Toffee Nut Latte', 'Chestnut Praline Latte', 
    'Cinnamon Roll Frappuccino', 'Smores Frappuccino', 'Vanilla Bean Frappuccino', 'Matcha Green Tea Latte', 
    'Pink Drink', 'Purple Drink', 'Orange Drink', 'Strawberry Acai Refresher', 'Very Berry Hibiscus Refresher', 
    'Dragon Drink', 'Mango Dragonfruit Refresher', 'Iced Pineapple Green Tea Infusion', 'Iced Peach Green Tea Lemonade', 
    'Iced Guava Passionfruit Drink', 'Iced Pineapple Matcha Drink', 'Iced Golden Ginger Drink', 
    'Iced Shaken Black Tea Lemonade', 'Iced Shaken Green Tea Lemonade', 'Iced Shaken Hibiscus Tea Lemonade', 
    'Iced Shaken Peach Green Tea Lemonade', 'Caramel Ribbon Crunch Frappuccino', 'Mocha Cookie Crumble Frappuccino', 
    'Double Chocolaty Chip Crème Frappuccino', 'Java Chip Frappuccino', 'Salted Caramel Mocha', 'Toasted White Chocolate Mocha', 
    'Chestnut Praline Frappuccino', 'Peppermint White Chocolate Mocha', 'Caramelized Honey Latte', 'Maple Pecan Latte', 
    'Cinnamon Shortbread Latte', 'Chestnut Praline Chai Tea Latte', 'Gingerbread Frappuccino', 
    'Cinnamon Shortbread Frappuccino', 'Caramel Apple Spice', 'Hot Caramel Macchiato', 'Hot Cinnamon Dolce Latte', 
    'Hot White Chocolate Mocha', 'Hot Peppermint Mocha', 'Iced Black Tea', 'Iced Green Tea', 
    'Iced Peach Green Tea', 'Iced Guava Black Tea', 'Iced Blueberry Black Tea', 'Iced Mango Black Tea', 
    'Iced Raspberry Black Tea', 'Iced Peach Green Tea', 'Iced Guava White Tea', 'Iced Strawberry Green Tea', 
    'Iced Black Tea Lemonade', 'Iced Green Tea Lemonade', 'Iced Peach Green Tea Lemonade', 'Iced Black Tea Latte', 
    'Iced Green Tea Latte', 'Iced London Fog Tea Latte', 'Iced Matcha Green Tea Latte', 'Caffè Misto', 'Cinnamon Dolce Latte', 
    'Cinnamon Dolce Crème', 'Pumpkin Spice Latte', 'Salted Caramel Mocha', 'Toasted White Chocolate Mocha', 'Toffee Nut Latte', 
    'Peppermint Mocha', 'Gingerbread Latte', 'Eggnog Latte', 'Maple Pecan Latte', 'Chestnut Praline Latte', 'Honey Almondmilk Cold Brew',
    'Toasted Coconut Cold Brew', 'Cocoa Cloud Macchiato', 'Caramel Cloud Macchiato', 'Iced Blonde Vanilla Latte', 'Iced Blonde Caffè Latte', 
    'Iced Blonde Caramel Cloud Macchiato', 'Iced Blonde Cinnamon Dolce Latte', 'Iced Blonde Vanilla Bean Coconutmilk Latte', 'Iced Caffè Mocha', 
    'Iced Toasted White Chocolate Mocha', 'Iced Toffee Nut Latte', 'Iced Cinnamon Dolce Latte', 'Iced Gingerbread Latte', 'Iced Pumpkin Spice Latte', 
    'Iced Salted Caramel Mocha', 'Iced Peppermint Mocha', 'Iced Eggnog Latte', 'Iced Maple Pecan Latte', 'Iced Chestnut Praline Latte', 'Iced Vanilla Latte', 
    'Iced Caramel Latte', 'Iced White Chocolate Mocha', 'Iced Caffè Americano with Milk', 'Iced Blonde Vanilla Bean Coconutmilk Latte', 
    'Iced Vanilla Bean Coconutmilk Latte', 'Iced Matcha Green Tea Latte', 'Iced Chai Tea Latte', 'Iced London Fog Tea Latte', 
    'Iced Pineapple Green Tea Infusion', 'Iced Peach Green Tea Infusion', 'Iced Black Tea Lemonade', 'Iced Peach Green Tea Lemonade',
    'Iced Guava White Tea Lemonade', 'Iced Pineapple Black Tea Infusion', 'Iced Peach Citrus White Tea Infusion', 'Iced Blueberry Black Tea Infusion',
    'Iced Passion Tango Tea Lemonade', 'Iced Golden Ginger Drink', 'Iced Coconutmilk Latte', 'Iced Cascara Latte', 'Iced Caramel Macchiato',
    'Iced Cinnamon Almondmilk Macchiato', 'Iced Vanilla Bean Coconutmilk Latte with Espresso', 'Iced Caffè Mocha with Cold Foam', 
    'Iced Blonde Vanilla Bean Coconutmilk Latte with Cold Foam', 'Iced Vanilla Bean Coconutmilk Latte with Cold Foam', 'Iced Pineapple Green Tea Infusion with Lemonade', 
    'Iced Guava White Tea Infusion with Lemonade', 'Iced Pineapple Black Tea Infusion with Lemonade', 'Iced Blueberry Black Tea Infusion with Lemonade', 
    'Iced Shaken Hibiscus Tea with Pomegranate Pearls', 'Iced Shaken Green Tea Lemonade with Pomegranate Pearls', 'Iced Pineapple Green Tea Infusion with Aloe', 
    'Iced Peach Citrus White Tea Infusion with Aloe', 'Iced Guava White Tea Infusion with Aloe', 'Iced Pineapple Black Tea Infusion with Aloe',
    'Iced Blueberry Black Tea Infusion with Aloe', 'Iced Shaken Hibiscus Tea with Coconutmilk', 'Iced Shaken Green Tea with Coconutmilk', 'Iced Shaken Blackberry Mojito Tea Lemonade',
    'Iced Shaken Peach Green Tea Lemonade', 'Teavana Shaken Iced Tea', 'Teavana Iced Green Tea','Teavana Iced Black Tea','Teavana Shaken Iced Peach Green Tea Lemonade',
    'Teavana Shaken Iced Pineapple Black Tea Infusion','Teavana Shaken Iced Peach Citrus White Tea Infusion','Teavana Shaken Iced Blueberry Black Tea Infusion',
    'Teavana Shaken Iced Strawberry Green Tea Infusion','Teavana Shaken Iced Mango Black Tea Infusion','Teavana Shaken Iced Guava White Tea Infusion',
    'Teavana Shaken Iced Hibiscus Tea with Pomegranate Pearls','Teavana Shaken Iced Peach Green Tea Infusion','Teavana Shaken Iced Blackberry Mojito Green Tea Infusion',
    'Teavana Shaken Iced Peach Citrus White Tea Infusion Lemonade','Teavana Shaken Iced Pineapple Green Tea Infusion','Teavana Shaken Iced Black Tea with Ruby Grapefruit & Honey',
    'Teavana Shaken Iced Mango Black Tea Infusion Lemonade','Teavana Shaken Iced Blueberry Black Tea Infusion Lemonade','Teavana Shaken Iced Guava White Tea Infusion Lemonade','Teavana Shaken Iced Pineapple Green Tea Infusion Lemonade']
]
